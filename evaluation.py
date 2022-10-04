#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 

import numpy as np


def interm_evaluation(model, weights_name, loaders, current_info, cnt):
    
    encoder = model[0]
    regressor = model[1]
    sp_regressor = model[2]

    MSE = nn.MSELoss()

    encoder_name = weights_name[0]
    regressor_name = weights_name[1]

    current_dir, current_time = current_info[0], current_info[1]

    count = cnt
    
    encoder.load_state_dict(torch.load(encoder_name), strict=False)
    regressor.load_state_dict(torch.load(regressor_name), strict=False)

    encoder.train(False)
    regressor.train(False)
    sp_regressor.train(False)

    total_loss = 0.0
    total_ccc_a, total_ccc_v = 0.0, 0.0
    total_rmse_a, total_rmse_v = 0.0, 0.0
    total_pcc_a, total_pcc_v = 0.0, 0.0
    cnt = 0

    use_gpu = torch.cuda.is_available()

    scores_list, labels_list = [], []
    with torch.no_grad():
        for batch_i, data_i in enumerate(loaders['val']):

            data, emotions = data_i['image'], data_i['va']
            valence = np.expand_dims(np.asarray(emotions[0]), axis=1)
            arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
            emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()

            if use_gpu:
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
            else:
                inputs, correct_labels = Variable(data), Variable(emotions)                                       
            
            z = encoder(inputs)
            scores, _ = regressor(z)

            scores_list.append(scores.detach().cpu().numpy())
            labels_list.append(correct_labels.detach().cpu().numpy())

            RMSE_valence = MSE(scores[:,0], correct_labels[:,0])**0.5
            RMSE_arousal = MSE(scores[:,1], correct_labels[:,1])**0.5
            
            total_rmse_v += RMSE_valence.item(); total_rmse_a += RMSE_arousal.item()
            cnt = cnt + 1

    scores_th = np.concatenate(scores_list, axis=0)
    labels_th = np.concatenate(labels_list, axis=0)

    std_l_v = np.std(labels_th[:,0]); std_p_v = np.std(scores_th[:,0])
    std_l_a = np.std(labels_th[:,1]); std_p_a = np.std(scores_th[:,1])

    mean_l_v = np.mean(labels_th[:,0]); mean_p_v = np.mean(scores_th[:,0])
    mean_l_a = np.mean(labels_th[:,1]); mean_p_a = np.mean(scores_th[:,1])

    PCC_v = np.cov(labels_th[:,0], np.transpose(scores_th[:,0])) / (std_l_v * std_p_v)
    PCC_a = np.cov(labels_th[:,1], np.transpose(scores_th[:,1])) / (std_l_a * std_p_a)
    CCC_v = 2.0 * np.cov(labels_th[:,0], np.transpose(scores_th[:,0])) / ( np.power(std_l_v,2) + np.power(std_p_v,2) + (mean_l_v - mean_p_v)**2 )
    CCC_a = 2.0 * np.cov(labels_th[:,1], np.transpose(scores_th[:,1])) / ( np.power(std_l_a,2) + np.power(std_p_a,2) + (mean_l_a - mean_p_a)**2 )

    sagr_v_cnt = 0
    for i in range(len(labels_th)):
        if np.sign(labels_th[i,0]) == np.sign(scores_th[i,0]) and labels_th[i,0] != 0:
            sagr_v_cnt += 1
    SAGR_v = sagr_v_cnt / len(labels_th)

    sagr_a_cnt = 0
    for i in range(len(labels_th)):
        if np.sign(labels_th[i,1]) == np.sign(scores_th[i,1]) and labels_th[i,1] != 0:
            sagr_a_cnt += 1
    SAGR_a = sagr_a_cnt / len(labels_th)

    final_rmse_v = total_rmse_v/cnt
    final_rmse_a = total_rmse_a/cnt

    # write results to log file
    with open(current_dir+'/log/'+current_time+'.txt', 'a') as f:
        f.writelines(['Itr: \t{},\n PCC: \t{}|\t {},\n CCC: \t{}|\t {},\n SAGR: \t{}|\t {},\n RMSE: \t{}|\t {}\n\n'.format(count, PCC_v[0,1], PCC_a[0,1], CCC_v[0,1], CCC_a[0,1], SAGR_v, SAGR_a, final_rmse_v, final_rmse_a)])
