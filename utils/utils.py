import os
import glob
import json
from map_elites import common as cm


def get_checkpoints(checkpoints_dir):
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'checkpoint_*'))
    return sorted(checkpoints)


def save_checkpoint(archive, n_evals, archive_name, checkpoints_dir, cfg):
    checkpoint_name = f'checkpoint_{n_evals:09d}/'
    filepath = os.path.join(checkpoints_dir, checkpoint_name)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    cm.save_archive(archive, n_evals, archive_name, save_path=filepath, save_models=True)
    save_cfg(cfg, filepath)


def cfg_dict(cfg):
    if isinstance(cfg, dict):
        return cfg
    else:
        return vars(cfg)


def save_cfg(cfg, save_path):
    cfg = cfg_dict(cfg)
    cfg_file = os.path.join(save_path, 'cfg.json')
    with open(cfg_file, 'w') as json_file:
        json.dump(cfg, json_file, indent=2)

