import logging
import torch
import glob
import re


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cuda')
        logging.info(f'load module from checkpoint: {last_ckpt_path}')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_episodes_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_episodes_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*episodes\_(\d+)\.ckpt', x)[0]))
