cd HiFAN
python -m torch.distributed.launch --nproc_per_node 1 main.py --run_mode train --config_exp ./configs/pascal_swinL.yml