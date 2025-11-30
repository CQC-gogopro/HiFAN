# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# pca代码

from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os

@torch.no_grad()
def test_phase(p, test_loader, model, criterion, epoch, save_edge=False):
    tasks = p.TASKS.NAMES

    performance_meter = PerformanceMeter(p, tasks)

    model.eval()

    if save_edge:
        tasks_to_save = ['edge']
        save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
        for save_dir in save_dirs.values():
            mkdir_if_missing(save_dir)
    
    for i, batch in enumerate(tqdm(test_loader)):  # dict_keys(['image', 'edge', 'human_parts', 'semseg', 'normals', 'sal', 'meta'])
        
        #******************获得真实图像大小begin******************
        # 获取图像的真实大小（去除padding）
        images = batch['semseg']  # torch.Size([6, 3, 512, 512])
        img_name = batch['meta']['img_name']
        batch_size, channels, height, width = images.shape
        real_size = []
        crop_index = []
        # 查找非零区域（真实图像区域）
        for b in range(batch_size):
            img = images[b]  # [1, 512, 512]
            # 将三个通道合并检查非零区域
            mask = img[0] !=255  # [512, 512]
            
            # 找到非零区域的边界
            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)
            
            if rows.any() and cols.any():
                # 找到非零区域的边界索引
                y_min, y_max = torch.where(rows)[0][[0, -1]]
                x_min, x_max = torch.where(cols)[0][[0, -1]]
                
                # 计算真实图像大小
                real_height = y_max - y_min + 1
                real_width = x_max - x_min + 1
                
                # 打印每张图片的真实大小
                # print(f"图片 {b} 的真实大小: {real_height}x{real_width}, 有效区域: [{y_min}:{y_max}, {x_min}:{x_max}]")
                real_size.append((int(real_height), int(real_width)))
                crop_index.append((int(y_min), int(y_max)+1, int(x_min), int(x_max)+1))
        info=dict(real_size=real_size, crop_index=crop_index, img_name=img_name)
        #******************获得真实图像大小end******************
        
        # Forward pass
        with torch.no_grad():
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

            output = model.module(images)
        
            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in tasks}, 
                                    {t: targets[t] for t in tasks})

            if save_edge:
                for task in tasks_to_save:
                    try:
                        save_model_pred_for_one_task(p, batch, output, save_dirs, task, epoch=epoch)
                    except:
                        pass


    eval_results = performance_meter.get_score(verbose = True)

    return eval_results