import argparse
import cv2
import os
import numpy as np
import sys
import torch
import pdb
import pprint
from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_test_dataset, get_train_dataloader, get_test_dataloader,\
                                get_optimizer, get_model, get_criterion
from utils.logger import Logger
from utils.train_utils import train_phase
from utils.test_utils import test_phase
from evaluation.evaluate_utils import PerformanceMeter

from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import time
start_time = time.time()

# DDP
import torch.distributed as dist
import datetime
dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(0, 3600*2))

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--local-rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--run_mode',
                    help='Config file for the experiment')
parser.add_argument('--trained_model', default=None,
                    help='Config file for the experiment')
parser.add_argument('--seed', default=0, type=int, help='')
parser.add_argument('--test_flops', default=0, type=int, help='')
parser.add_argument('--debug', default=False, type=bool, help='')
args = parser.parse_args()

print('local rank: %s' %args.local_rank)
torch.cuda.set_device(args.local_rank)

# CUDNN
torch.backends.cudnn.benchmark = True
# opencv
cv2.setNumThreads(0)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(args.seed)
    # Retrieve config file
    params = {'run_mode': args.run_mode}
    p = create_config(args.config_exp, params)
    if args.local_rank == 0:
        sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
        pprint.pprint(p)
        pprint.pprint(args)
        print("seed:", args.seed)

    # tensorboard
    tb_log_dir = p.root_dir + '/tb_dir' #os.path.join(p['output_dir'], 'tensorboard_logdir')
    p.tb_log_dir = tb_log_dir
    if args.local_rank == 0:
        train_tb_log_dir = tb_log_dir + '/train'
        test_tb_log_dir = tb_log_dir + '/test'
        tb_writer_train = SummaryWriter(train_tb_log_dir)
        tb_writer_test = SummaryWriter(test_tb_log_dir)
        if args.run_mode == 'train':
            mkdir_if_missing(tb_log_dir)
            mkdir_if_missing(train_tb_log_dir)
            mkdir_if_missing(test_tb_log_dir)
        if args.run_mode == 'train' and not args.debug:
            import shutil
            models_dir = './models'
            dst_models_dir = os.path.join(p.root_dir, 'models')
            if os.path.exists(models_dir):
                shutil.copytree(models_dir, dst_models_dir, dirs_exist_ok=True)
            config_dir = './configs'
            dst_configs_dir = os.path.join(p.root_dir, 'configs')
            if os.path.exists(models_dir):
                shutil.copytree(config_dir, dst_configs_dir, dirs_exist_ok=True)
        print(f"Tensorboard dir: {tb_log_dir}")
    else:
        tb_writer_train = None
        tb_writer_test = None


    # Get model
    model = get_model(p)
    num = 0
    for mp in model.parameters():
        num += mp.numel()
    print(f'Model Param Size: {num}')
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)    # 多卡同步的BN
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Get criterion
    criterion = get_criterion(p).cuda()

    # Optimizer
    scheduler, optimizer = get_optimizer(p, model)
    
    if 0:
        import matplotlib.pyplot  as plt 
        # 假设 num_epochs 和 scheduler 已经定义 
        lr_history = []
        for epoch in range(40000):
            lr_history.append(optimizer.param_groups[0]['lr']) 
            scheduler.step() 
        
        # 绘制学习率变化图 
        plt.plot(lr_history) 
        plt.xlabel('Epoch') 
        plt.ylabel('Learning  Rate')
        plt.title('Learning  Rate Schedule')
        
        # 保存图像到本地 
        plt.savefig('schedule.png')

    # Performance meter init
    performance_meter = PerformanceMeter(p, p.TASKS.NAMES)

    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    if args.run_mode == 'train':
        train_dataset = get_train_dataset(p, train_transforms)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        train_dataloader = get_train_dataloader(p, train_dataset, train_sampler)
        test_dataset = get_test_dataset(p, val_transforms)
        test_dataloader = get_test_dataloader(p, test_dataset)
    elif args.run_mode == 'infer':
        test_dataset = get_test_dataset(p, val_transforms)
        test_dataloader = get_test_dataloader(p, test_dataset)

    
    # Resume from checkpoint
    if os.path.exists(p['checkpoint']) or args.run_mode in ['infer']:
        if args.trained_model != None:
            checkpoint_path = args.trained_model
        else:
            checkpoint_path = p['checkpoint']
        if args.local_rank == 0:
            print('Use checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint.keys():
            start_epoch = checkpoint['epoch'] + 1 # epoch count is not used
        else:
            start_epoch = 0
        if 'iter_count' in checkpoint.keys():
            iter_count  = checkpoint['iter_count'] # already + 1 when saving
            forward_count = int(checkpoint['iter_count']*p.ACCUMULATION_STEPS)
        else:
            iter_count = 0
            forward_count = 0
        best_imp = -100   #********************************
        
    else:
        if args.local_rank == 0:
            print('Fresh start...')
        start_epoch = 0
        iter_count = 0
        best_imp = -100
        forward_count = 0
    if DEBUG_FLAG and args.local_rank == 0:
        print("\nFirst Testing...")
        if True:
            eval_test = test_phase(p, test_dataloader, model, criterion, iter_count)
        else:
            eval_test = {}
        print(eval_test)
    
    # Train loop
    if args.run_mode == 'train':
        for epoch in range(start_epoch, p['epochs']):
            train_sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print('Epoch %d/%d' %(epoch+1, p['epochs']))
                print('-'*10)

            end_signal, iter_count, forward_count, best_imp = train_phase(p, args, train_dataloader, test_dataloader, model, criterion, 
                                                optimizer, scheduler, epoch, tb_writer_train, tb_writer_test, iter_count, forward_count, best_imp)

            if end_signal:
                break

    # running eval
    if args.local_rank == 0:
        eval_epoch = iter_count # start_epoch
        eval_test = test_phase(p, test_dataloader, model, criterion, eval_epoch, save_edge=True)
        print('Infer test restuls:')
        print(eval_test)

        end_time = time.time()
        run_time = (end_time-start_time) / 3600
        print('Total running time: {} h.'.format(run_time))

if __name__ == "__main__":
    # IMPORTANT VARIABLES
    DEBUG_FLAG = False # When True, test the evaluation code when started

    assert args.run_mode in ['train', 'infer']
    main()
