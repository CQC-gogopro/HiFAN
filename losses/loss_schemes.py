#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt, tasks, t_iter=0, **kwargs):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))

        if self.p.intermediate_supervision:
            inter_preds = pred['inter_preds']
            losses_inter = {t: self.loss_ft[t](inter_preds[t], gt[t]) for t in self.tasks}
            for k, v in losses_inter.items():
                out['inter_%s' %(k)] = v
                out['total'] += self.loss_weights[k] * v #* 0.5

        return out
    
class MultiTaskLoss_linear_decay(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict, decays: dict, inter_weight=1):
        super(MultiTaskLoss_linear_decay, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.decays = decays
        self.inter_weight = inter_weight
        

    def forward(self, pred, gt, tasks, t_iter=0, **kwargs):
        # t_iter: 0-1
        # decays 训练从1到1-decays
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] *(1-self.decays[t]*t_iter) * out[t] for t in tasks]))

        if self.p.intermediate_supervision:
            inter_preds = pred['inter_preds']
            losses_inter = {t: self.loss_ft[t](inter_preds[t], gt[t]) for t in self.tasks}
            for k, v in losses_inter.items():
                out['inter_%s' %(k)] = v
                out['total'] += self.inter_weight *self.loss_weights[k] *(1-self.decays[k]*t_iter) * v #* 0.5

        return out

class MultiTaskLoss_log_linear_decay(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict, decays: dict):
        super(MultiTaskLoss_log_linear_decay, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        # assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.decays = decays
        # decays 训练从1到decays


    def forward(self, pred, gt, tasks, t_iter=0, **kwargs):
        # t_iter: 0-1
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * (1 - (1-self.decays[t])*t_iter) * torch.log(out[t]+1e-8) for t in tasks]))

        if self.p.intermediate_supervision:
            inter_preds = pred['inter_preds']
            losses_inter = {t: self.loss_ft[t](inter_preds[t], gt[t]) for t in self.tasks}
            for k, v in losses_inter.items():
                out['inter_%s' %(k)] = v
                out['total'] += self.loss_weights[k] * (1 - (1-self.decays[k])*t_iter) * torch.log(v+1e-8) #* 0.5

        return out
    

class MultiTaskLoss_log(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss_log, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        if 'aux_loss_weight' in p:
            self.aux_loss_weight = p['aux_loss_weight']
        else: 
            self.aux_loss_weight = 1.0

    
    def forward(self, pred, gt, tasks, t_iter=0, **kwargs):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * torch.log(out[t]+1e-8) for t in tasks]))

        if self.p.intermediate_supervision:
            inter_preds = pred['inter_preds']
            losses_inter = {t: self.aux_loss_weight * self.loss_ft[t](inter_preds[t], gt[t]) for t in self.tasks}
            for k, v in losses_inter.items():
                out['inter_%s' %(k)] = v
                out['total'] += self.loss_weights[k] * torch.log(v+1e-8) #* 0.5

        return out