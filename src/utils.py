import torch
import torch.nn as nn

from torch.autograd import Function, Variable


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)
        m.bias.data.fill_(0.01)


def vector_difference(x1, x2):
    x1_n = x1 / (torch.norm(x1, p=2, dim=1, keepdim=True)+1e-6)
    x2_n = x2 / (torch.norm(x2, p=2, dim=1, keepdim=True)+1e-6)
    return x1_n - x2_n


def pcc_ccc_loss(labels_th, scores_th):
    std_l_v = torch.std(labels_th[:,0]); std_p_v = torch.std(scores_th[:,0])
    std_l_a = torch.std(labels_th[:,1]); std_p_a = torch.std(scores_th[:,1])
    mean_l_v = torch.mean(labels_th[:,0]); mean_p_v = torch.mean(scores_th[:,0])
    mean_l_a = torch.mean(labels_th[:,1]); mean_p_a = torch.mean(scores_th[:,1])
    
    PCC_v = torch.mean( (labels_th[:,0] - mean_l_v) * (scores_th[:,0] - mean_p_v) ) / (std_l_v * std_p_v)
    PCC_a = torch.mean( (labels_th[:,1] - mean_l_a) * (scores_th[:,1] - mean_p_a) ) / (std_l_a * std_p_a)
    CCC_v = (2.0 * std_l_v * std_p_v * PCC_v) / ( std_l_v.pow(2) + std_p_v.pow(2) + (mean_l_v-mean_p_v).pow(2) )
    CCC_a = (2.0 * std_l_a * std_p_a * PCC_a) / ( std_l_a.pow(2) + std_p_a.pow(2) + (mean_l_a-mean_p_a).pow(2) )
    
    PCC_loss = 1.0 - (PCC_v + PCC_a)/2
    CCC_loss = 1.0 - (CCC_v + CCC_a)/2
    return PCC_loss, CCC_loss, CCC_v, CCC_a


def pair_mining(inp1, inp2, fixed_sample, is_positive):
    sim_matrix = inp1 @ inp2.t()

    if fixed_sample:  # Ranking-based mining
        batch_size = inp2.size(0)
        sort_size = int(0.8 * batch_size)

        if is_positive:
            _, ind = torch.topk(sim_matrix, k=sort_size, dim=0, largest=True)
        else:
            _, ind = torch.topk(sim_matrix, k=sort_size, dim=0, largest=False)
    
        sort_ind = ind[:,0]
        inp1_sort = inp1[sort_ind]
        inp2_sort = inp2[sort_ind]
        return inp1_sort, inp2_sort
    else:  
        # Pair mining from metric learning.
        # Because of absense of categorical label, we slightly tuned original pair mining.
        epsilon = 0.0005
        if is_positive:
            value, _ = torch.max(sim_matrix, dim=0)
            ind = sim_matrix < (value-epsilon)
        else:
            value, _ = torch.min(sim_matrix, dim=0)
            ind = sim_matrix > (value+epsilon)

        sort_ind = []
        for i in range(len(ind[0])):
            prob = sum(ind[i]) / len(ind[i])
            # Select informative pairs for constructing positive or negative pairs as probability values.
            if prob >= 0.7:
                sort_ind.append(i)

        inp1_sort = inp1[sort_ind]
        inp2_sort = inp2[sort_ind]
        return inp1_sort, inp2_sort


def penalty_function(inp1, inp2):
    return 0.5 * torch.ones_like(inp1[:,0]) * (torch.sign(inp1[:,0]) != torch.sign(inp2[:,0]))
