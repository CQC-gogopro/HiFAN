# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os, json
from evaluation.evaluate_utils import PerformanceMeter, count_improvement
from utils.utils import to_cuda
import torch
from tqdm import tqdm
from utils.test_utils import test_phase
import pdb
# from fvcore.nn import flop_count
from .flop_count import count_flops

# import torch.profiler

def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, 
                scheduler, epoch, tb_writer, tb_writer_test, iter_count, forward_count, best_imp):
    """ Vanilla training with fixed loss weights """
    model.train() 

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        # Forward pass
        batch = to_cuda(cpu_batch)
        images = batch['image'] 
        
        output = model(images)    
        if 0 and i==1:
            # test flops
            model.eval() 
            images = images[0].unsqueeze(0)
            flops = count_flops(model, (images,))
            model.train() 
            print(f"Total GFLOPs: {flops}")
            exit()
         
        forward_count += 1
        # Measure loss
        t_iter = iter_count/p['max_iter']
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES, t_iter=t_iter)
        
        # get learning rate
        if scheduler is not None:
            lr = scheduler.get_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']
        loss_dict['lr'] = torch.tensor(lr)
        loss_dict['total'].backward()

        #TODO 累计梯度
        if (forward_count+1) % p.ACCUMULATION_STEPS == 0:  # 每 ACCUMULATION_STEPS 次更新参数（梯度累计）
            if tb_writer is not None:
                update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)

            if args.local_rank == 0:
                print(f'Iter {iter_count}, ', end="")
                for k, v in loss_dict.items():
                    print('{}: {:.7f} | '.format(k, v), end="")
                print()
            
            iter_count += 1
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
            except:
                pass
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            
            # end condition
            if iter_count >= p.max_iter:
                print('Max itereaction achieved.')
                if args.local_rank == 0:
                    curr_result = test_phase(p, test_dataloader, model, criterion, epoch)
                    torch.save({'model': model.state_dict()}, p['checkpoint'].replace('checkpoint.pth.tar', 'last_model.pth.tar'))
                    with open(p['checkpoint'].replace('checkpoint.pth.tar', 'last.txt'), 'w') as f:
                        json.dump(curr_result, f, indent=4)
                end_signal = True
                return True, iter_count, forward_count, best_imp
            else:
                end_signal = False

    # Perform evaluation
    begin_eva = 1
    if args.local_rank == 0 and epoch >= begin_eva:
        print('Evaluate at epoch {}'.format(epoch))
        curr_result = test_phase(p, test_dataloader, model, criterion, epoch)
        # tb_update_perf(p, tb_writer_test, curr_result, iter_count)
        print('Evaluate results at epoch {}: \n'.format(epoch))
        print(curr_result)
        with open(os.path.join(p['save_dir'], p.version_name + '_' + str(epoch) + '.txt'), 'w') as f:
            json.dump(curr_result, f, indent=4)

        current_imp = count_improvement(p['train_db_name'], curr_result, p['TASKS']['NAMES'])
        if current_imp > best_imp:
            best_imp = current_imp
            # Checkpoint after evaluation
            print('Checkpoint starts at epoch {}....'.format(epoch))
            torch.save({'model': model.state_dict()}, p['checkpoint'])
            print('Checkpoint finishs.')
            
            curr_result.update({'epoch': epoch, 'best_imp': best_imp})
            with open(p['checkpoint'].replace('checkpoint.pth.tar', 'best.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)
        model.train() # set model back to train status

        # if end_signal:
        #     return True, iter_count

    return False, iter_count, forward_count, best_imp


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, warmup_start_lr=0, warmup_mode='linear', last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.warmup_mode = warmup_mode
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            if self.warmup_mode == 'linear':
                return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
            elif self.warmup_mode == 'constant':
                return [self.warmup_start_lr for _ in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]

class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + torch.cos(torch.tensor(self.last_epoch / self.T_max * 3.141592653589793))) / 2 for base_lr in self.base_lrs]

def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)
