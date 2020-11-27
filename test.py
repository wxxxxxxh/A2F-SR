''' Baisc packages
'''
import os
import glob
import tqdm
import copy
import random
import importlib
import numpy as np
from decimal import Decimal
from collections import OrderedDict

''' Configuration packages
'''
import yaml
import argparse
from easydict import EasyDict as edict

''' PyTorch packages
'''
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

''' Customized packages
'''
from utility import optimizer_util as opti_util
from Data_zoo import common_utils as cutils

_YAML_MAP = {
    'AAF_S_x4' :'./yaml/train_s_x4.yaml',
    'AAF_SD_x4':'./yaml/train_sd_x4.yaml',
    'AAF_M_x4' :'./yaml/train_m_x4.yaml',
    'AAF_L_x4' :'./yaml/train_l_x4.yaml',
    'AAF_S_x3' :'./yaml/train_s_x3.yaml',
    'AAF_SD_x3':'./yaml/train_sd_x3.yaml',
    'AAF_M_x3' :'./yaml/train_m_x3.yaml',
    'AAF_L_x3' :'./yaml/train_l_x3.yaml',
    'AAF_S_x2' :'./yaml/train_s_x2.yaml',
    'AAF_SD_x2':'./yaml/train_sd_x2.yaml',
    'AAF_M_x2' :'./yaml/train_m_x2.yaml',
    'AAF_L_x2' :'./yaml/train_l_x2.yaml',
}

def net_test(args, cfg):
    ''' 0. Import model from Model_zoo
	'''
    AAF = importlib.import_module('.{}'.format(args.model.name.lower()), package='Model_zoo')
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    ''' 1. Model
    '''
    print('========> Build model')
    
    SRmodel = AAF.create_model(args.training)
    pretrained_model = os.path.join(args.ckp.save_root, cfg.config, args.testing.pretrained_model_path)
    SRmodel.model.load_state_dict(torch.load(pretrained_model))
    SRmodel.eval()
    for k,v in SRmodel.model.named_parameters():
        v.requires_grad = False
    SRmodel = SRmodel.to(device)
    number_parameters = sum(map(lambda x: x.numel(), SRmodel.parameters()))
    print('========> Parameters: {}'.format(number_parameters))

    ''' 2.Data
    '''
    dir_lr = os.path.join(args.testing.test_dir)
    dir_lr_list = cutils.get_image_paths(dir_lr)
    dir_sr = os.path.join(args.testing.result_dir, cfg.config)
    if not os.path.exists(dir_sr):
        os.makedirs(dir_sr, exist_ok=True)

    ''' 3.Test
    '''
    idx = 0
    test_results = OrderedDict()
    test_results['runtime'] = []
    


    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    for img in dir_lr_list:
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        print('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        lr = cutils.imread_uint(img, n_channels=3)
        lr = cutils.uint2tensor4(lr)
        lr = lr.float()
        lr = lr.to(device)
        
        t_start.record()
        sr = SRmodel(lr)
        t_end.record()
        torch.cuda.synchronize()
        sr = cutils.quantize(sr, args.training.rgb_range)

        cutils.imsave(sr, os.path.join(args.testing.result_dir, cfg.config, img_name+ext))
        test_results['runtime'].append(t_start.elapsed_time(t_end))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    print(('------> Average runtime is : {:.6f} seconds'.format(ave_runtime)))


if __name__ == "__main__":
    ''' Parameters
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='AAF_S_x4')
    cfg = parser.parse_args()
    yaml_path = _YAML_MAP['{}'.format(cfg.config)]
    args = edict(yaml.load(open(yaml_path, 'r')))
    print(args.DESCRIPTION)
    cudnn.benchmark = args.cudnn_benchmark

    ''' Testing
    '''
    print('LR_dir -> {}'.format(os.path.join(args.testing.test_dir)))
    print('Load_ckp -> {}'.format(args.testing.pretrained_model_path))

    print('--------------------------Begin-----------------------')
    net_test(args, cfg)