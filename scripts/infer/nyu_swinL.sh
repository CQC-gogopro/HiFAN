cd HiFAN
python -m torch.distributed.launch --nproc_per_node 1 main.py --run_mode infer --trained_model ./outputs/nyu_swinL/checkpoint.pth.tar --config_exp ./configs/nyu_swinL.yml