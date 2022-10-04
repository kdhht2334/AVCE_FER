#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os

import argparse
app = argparse.ArgumentParser()
app.add_argument("--freq", type=int, default=250, help='Saving frequency.')
app.add_argument("--model", type=str, default='alexnet', help='alexnet / resnet18 / resnet50 / resnet101.')
app.add_argument("--online_tracker", type=int, default=1, help='Whether or not applying wandb.')
app.add_argument("--dataset", type=str, default='aff_wild', help='aff_wild / aff_wild2 / afew_va / affectnet.')

app.add_argument("--data_path", type=str, default='/')
app.add_argument("--save_path", type=str, default='/')
args = app.parse_args()

import torch
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from time import gmtime, strftime
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.models as models
import torchvision.datasets as datasets

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F 

from fabulous.color import fg256

import cvxpy as cp
import wandb

from utils import init_weights, pair_mining, pcc_ccc_loss, vector_difference, penalty_function
from dataset_utils import FaceDataset
from models import encoder_Alex, encoder_R18, regressor_Alex, regressor_R18, regressor_R50, regressor_R101, spregressor, vregressor
from cvx_utils import OptLayer
from evaluation import interm_evaluation


def model_training(model, optimizer, scheduler, current_info, num_epochs):

    if args.online_tracker:
        wandb.init(project="AVCE_FER")

    cnt, balance_factor, margin = 0, 0.8, 1.0

    encoder = model[0]
    regressor = model[1]
    sp_regressor = model[2]
    sp_regressor.train(True)
    sp_regressor.apply(init_weights)

    v_regressor = model[3]
    v_regressor.train(True)
    v_regressor.apply(init_weights)
    
    ALPHA = Variable(torch.FloatTensor([0.5]).cuda(), requires_grad=True)
    BETA  = Variable(torch.FloatTensor([0.005]).cuda(), requires_grad=True)
    GAMMA = Variable(torch.FloatTensor([0.5]).cuda(), requires_grad=True)

    enc_opt = optimizer[0]
    reg_opt = optimizer[1]
    spreg_opt = optimizer[2]
    vreg_opt = optimizer[3]

    MSE = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # SparseMax
    z = cp.Variable(32)
    x = cp.Parameter(32)

    f_ = lambda z,x : cp.sum_squares(z - x) if isinstance(z, cp.Variable) else torch.sum((x-z)**2)
    g_ = lambda z,x : -z
    h_ = lambda z,x: cp.sum(z) - 1 if isinstance(z, cp.Variable) else z.sum() - 1
    sp_layer = OptLayer([z], [x], f_, [g_], [h_])

    # SoftMax
    zs = cp.Variable(32)
    xs = cp.Parameter(32)

    fs_ = lambda zs,xs: -zs@xs - cp.sum(cp.entr(zs)) if isinstance(zs, cp.Variable) else -zs@xs + zs@torch.log(zs)
    hs_ = lambda zs,xs: cp.sum(zs) - 1 if isinstance(zs, cp.Variable) else zs.sum() - 1
    sm_layer = OptLayer([zs], [xs], fs_, [], [hs_])

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Itr per epoch
    for epoch in range(num_epochs):

        print('epoch ' + str(epoch) + '/' + str(num_epochs-1))
        epoch_iterator = tqdm(loaders['train'],
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for batch_i, data_i in enumerate(epoch_iterator):

            for enc_param_group in enc_opt.param_groups:
                aa = enc_param_group['lr']
            for reg_param_group in reg_opt.param_groups:
                bb = reg_param_group['lr']
            
            data, emotions = data_i['image'], data_i['va']
            valence = np.expand_dims(np.asarray(emotions[0]), axis=1)
            arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
            emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()

            if use_gpu:
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
            else:
                inputs, correct_labels = Variable(data), Variable(emotions)

            # ---------------
            # Train regressor
            # ---------------
            z   = encoder(inputs)
            scores, z_btl = regressor(z)

            z_sp_btl = sp_layer(z_btl.cpu()).cuda()
            z_sm_btl = F.softmax(z_btl)
            sp_scores = sp_regressor(z_sp_btl)
            sm_scores = sp_regressor(z_sm_btl)

#            scores_pm, sp_scores = pair_mining(scores, sp_scores, fixed_sample=0, is_positive=1)
#            scores_nm, sm_scores = pair_mining(scores, sm_scores, fixed_sample=0, is_positive=0)

            sp_scores_norm = torch.norm(sp_scores, p=2, dim=1)
            sm_scores_norm = torch.norm(sm_scores, p=2, dim=1)
            scores_norm = torch.norm(scores, p=2, dim=1)
            diff_norm_pos = balance_factor * ( 1. - torch.abs(sp_scores_norm - scores_norm) )
            diff_norm_neg = balance_factor * ( 1. - torch.abs(sm_scores_norm - scores_norm) )

            inner_product = (sp_scores * scores).sum(dim=1)
            a_norm = sp_scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
            b_norm = scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
            cos_sp = inner_product / (2 * a_norm * b_norm)
            _angle_sp = torch.acos(cos_sp)
            angle_sp = torch.clamp(1. - 0.25 * _angle_sp/math.pi, min=1e-6, max=1.)
            with torch.no_grad():
                cos_sp_mean = _angle_sp.mean()
                angle_sp_mean = angle_sp.mean()

            inner_product = (sm_scores * scores).sum(dim=1)
            a_norm = sm_scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
            b_norm = scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
            cos_sm = inner_product / (2 * a_norm * b_norm)
            _angle_sm = torch.acos(cos_sm)
            angle_sm = torch.clamp(1. - 0.25 * _angle_sm/math.pi, min=1e-6, max=1.)
            with torch.no_grad():
                cos_sm_mean = _angle_sm.mean()
                angle_sm_mean = angle_sm.mean()

            pos_sim_func = torch.unsqueeze(angle_sp+diff_norm_pos, dim=1) - 0.1 * penalty_function(sp_scores, scores)
            neg_sim_func = torch.unsqueeze(angle_sm+diff_norm_neg, dim=1) - 0.1 * penalty_function(sm_scores, scores)

            AVCE_scores = 0.
            for i in range(pos_sim_func.size(0)):
                AVCE_scores += pos_sim_func[i] - ALPHA * neg_sim_func.mean() \
                    -0.5 * BETA * pos_sim_func[i].pow(2) - 0.5 * GAMMA * neg_sim_func.pow(2).mean()
            AVCE = AVCE_scores.mean(); del AVCE_scores


            pcc_loss, ccc_loss, ccc_v, ccc_a = pcc_ccc_loss(correct_labels, scores)

            MSE_v = MSE(scores[:,0], correct_labels[:,0])
            MSE_a = MSE(scores[:,1], correct_labels[:,1])

            enc_opt.zero_grad()
            reg_opt.zero_grad()
            spreg_opt.zero_grad()
            loss = (MSE_v + MSE_a) - 1e-4 * AVCE.cuda() + (0.5 * pcc_loss + 0.5 * ccc_loss)
            loss.backward(retain_graph=True)

            enc_opt.step()
            reg_opt.step()
            spreg_opt.step()

            
            ### Metric-based regularization ###
            enc_opt.zero_grad()
            reg_opt.zero_grad()
            vreg_opt.zero_grad()

            z = encoder(inputs)
            _, z_btl = regressor(z)
            z_sp_btl = sp_layer(z_btl.cpu())
            z_sm_btl = F.softmax(z_btl)

            v_btl    = v_regressor(z_btl)
            v_sp_btl = v_regressor(z_sp_btl.cuda())
            v_sm_btl = v_regressor(z_sm_btl)

#            v_btl_pm, v_sp_btl = pair_mining(v_btl, v_sp_btl, fixed_sample=0, is_positive=1)
#            v_btl_nm, v_sm_btl = pair_mining(v_btl, v_sm_btl, fixed_sample=0, is_positive=0)

            holding_vector1 = torch.norm(vector_difference(v_sp_btl,v_btl), p=2, dim=1, keepdim=True)
            holding_vector2 = torch.norm(vector_difference(v_sm_btl,v_btl), p=2, dim=1, keepdim=True)
            one_vector = torch.ones_like(holding_vector1)

            only_pushing_loss = torch.mean(F.relu(margin - torch.norm(v_sp_btl - v_sm_btl, p=2, dim=1).pow(2)))
            reg_loss = only_pushing_loss.cuda() + 0.01 * (torch.mean(holding_vector1 - one_vector) + torch.mean(holding_vector2 - one_vector)).cuda()
            reg_loss.backward()

            enc_opt.step()
            reg_opt.step()
            vreg_opt.step()

            
            if args.online_tracker:
                wandb.log({
                    "loss": loss.item(), 
                    'RPC': AVCE.item(),
                    "Enc_lr": aa, "Reg_lr": bb,
                    "epoch": epoch, "ccc_v": ccc_v.item(), "ccc_a": ccc_a.item(),
                    "MSE (v)": MSE_v, "MSE (a)": MSE_a
                })

            if cnt % args.freq == 0 or cnt == 50:
                encoder_name   = args.save_path + 'Enc_{}_{}.t7'.format(cnt, epoch)
                regressor_name = args.save_path + 'Reg_{}_{}.t7'.format(cnt, epoch)
                torch.save(encoder.state_dict(), encoder_name)
                torch.save(regressor.state_dict(), regressor_name)

                # Validation
                interm_evaluation([encoder, regressor, sp_regressor], [encoder_name, regressor_name], 
                                  loaders, current_info, cnt)

            cnt = cnt + 1

            scheduler[0].step()
            scheduler[1].step()
            scheduler[2].step()
            scheduler[3].step()


if __name__ == "__main__":

    # ----------
    # Initialize
    # ----------
    import warnings
    warnings.filterwarnings("ignore")
    
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_dir  = os.getcwd()
    current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

    # Make log file
    if args.online_tracker:
        with open(current_dir+'/log/'+current_time+'.txt', 'w') as f:
            f.writelines(["Title: AVCE (Model: {}\t Dataset: {}).\n".format(args.model, args.dataset)])

    #------------
    # Data loader
    #------------
    training_path = args.data_path + 'training_list.csv'
    validation_path = args.data_path + 'validation_list.csv'

    face_dataset = FaceDataset(csv_file=training_path,
                               root_dir=args.data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(256), transforms.RandomCrop(size=224),
                                   transforms.ColorJitter(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                               ]), inFolder=None)

    face_dataset_val = FaceDataset(csv_file=validation_path,
                                   root_dir=args.data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(256), transforms.CenterCrop(size=224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                   ]), inFolder=None)

    
    if args.model == 'alexnet': batch_size = 256
    elif args.model == 'resnet18': batch_size = 128
    else: batch_size = 64

    dataloader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(face_dataset_val, batch_size=64, shuffle=False)
    
    loaders = {'train': dataloader, 'val': dataloader_val}
    
    use_gpu = torch.cuda.is_available()
    dataset_size = {'train': len(face_dataset), 'val': len(face_dataset_val)}
    train_size = dataset_size['train']; val_size = dataset_size['val']
    print(fg256("yellow", 'train | val size: {} | {}'.format(train_size, val_size)))

    #----------------
    # Build DNN model
    #----------------
    if args.model == 'alexnet':
        encoder2     = encoder_Alex().cuda()
        regressor    = regressor_Alex().cuda()
    elif args.model == 'resnet18':
        encoder2     = encoder_R18().cuda()
        regressor    = regressor_R18().cuda()
    elif args.model == 'resnet50':
        print(fg256("green", 'Choose model:ResNet50'))
        import pretrainedmodels
        resnet50     = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained=None)
        encoder2     = nn.DataParallel(resnet50).to(device)
        regressor    = regressor_R50().to(device)
    elif args.model == 'resnet101':
        print(fg256("green", 'Choose model:ResNet101'))
        import pretrainedmodels
        resnet101    = pretrainedmodels.__dict__['resnet101'](num_classes=1000, pretrained=None)
        encoder2     = nn.DataParallel(resnet101).to(device)
        regressor    = regressor_R101().to(device)

    sp_regressor = spregressor(discrete_opt=0).to(device)
    v_regressor  = vregressor().to(device)

    enc_opt   = optim.Adam(encoder2.parameters(),    lr = 1e-4, betas = (0.5, 0.9))
    reg_opt   = optim.Adam(regressor.parameters(),   lr = 1e-4, betas = (0.5, 0.9))
    spreg_opt = optim.SGD(sp_regressor.parameters(), lr = 1e-2, momentum=0.9)
    vreg_opt  = optim.SGD(v_regressor.parameters(),  lr = 1e-2, momentum=0.9)
    
    enc_exp_lr_scheduler   = lr_scheduler.MultiStepLR(enc_opt, milestones=[5e3,25e3,45e3,65e3,85e3], gamma=0.8)
    reg_exp_lr_scheduler   = lr_scheduler.MultiStepLR(reg_opt, milestones=[5e3,25e3,45e3,65e3,85e3], gamma=0.8)
    spreg_exp_lr_scheduler = lr_scheduler.MultiStepLR(spreg_opt, milestones=[5e3,25e3,45e3,65e3,85e3], gamma=0.8)
    vreg_exp_lr_scheduler  = lr_scheduler.MultiStepLR(vreg_opt, milestones=[5e3,25e3,45e3,65e3,85e3], gamma=0.8)
    
    #-----------------------
    # Training or evaluation
    #-----------------------
    model_training([encoder2             , regressor            , sp_regressor           , v_regressor]           ,
                   [enc_opt              , reg_opt              , spreg_opt              , vreg_opt]              ,
                   [enc_exp_lr_scheduler , reg_exp_lr_scheduler , spreg_exp_lr_scheduler , vreg_exp_lr_scheduler] ,
                   [current_dir, current_time], num_epochs=100)
