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
from tensorboardX import SummaryWriter

''' Customized packages
'''
from Data_zoo import div2k
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

def net_train(args, cfg):
    ''' 0. Import model from Model_zoo
	'''
    AAF = importlib.import_module('.{}'.format(args.model.name.lower()), package=args.model.zoo)



    ''' 1. Load data and return a dataloader
    '''
    print('========> Load data!')
    args_data = args.dataset
    train_set = div2k.create_dataset(args_data)
    eval_set  = div2k.create_dataset(args_data, phase='eval')
    train_loader = DataLoader(train_set, batch_size=args_data.batch_size, num_workers=args_data.num_workers, shuffle=True, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=1, num_workers=1, shuffle=True, pin_memory=True)

    ''' 2. Loss and logger
    '''
    print('========> Build loss fuction!')
    Loss = nn.L1Loss()
    writer = SummaryWriter(args.tensorboard.save_path)


    ''' 3. Model
    '''
    print('========> Build model')
    SRmodel = AAF.create_model(args.training)
    assert args.n_GPUs>0
    GPU_list = [i for i in range(args.n_GPUs)]
    device = torch.device('cuda')
    SRmodel = nn.DataParallel(SRmodel, device_ids=GPU_list)
    SRmodel.to(device)


    ''' 4. Load checkpoint
    '''
    if args.ckp.load_ckp:
        if args.ckp.type==0:
            ep = 'latest.pt'
        elif args.ckp.type==1:
            ep = 'best.pt'
        else:
            ep = '{}.pt'.format(args.ckp.type)
        print('========> Load checkpoint from {}!'.format(ep))
        ckp_file = os.path.join(args.ckp.save_root, args.ckp.ckp_path+ep)
        state_d = torch.load(ckp_file)
        SRmodel.module.model.load_state_dict(state_d)

    ''' 5. get number of parameters
    '''
    number_parameters = sum(map(lambda x: x.numel(), SRmodel.parameters()))

    ''' 6. Build optimizer
    '''
    print('========> Build optimizer and schedualr!')
    optimizer = opti_util.make_optimizer(args.optimizer, SRmodel.module)
    schedular = opti_util.make_scheduler(args.schedular, optimizer)



    ''' Train model
    '''
    print('========> Parameters: {}'.format(number_parameters))
    print('========> Start to train!')
    for epoch in range(args.training.epochs):
        bar = tqdm.tqdm(train_loader)
        l_rate = schedular.get_lr()[0]
        SRmodel.train()
        loss_list = []

        for batch, (hr, lr, _) in enumerate(bar):
            hr = hr.to(device)
            lr = lr.to(device)

            optimizer.zero_grad()
            sr = SRmodel(lr)

            loss_val = Loss(sr, hr)
            loss_val.backward()
            optimizer.step()
            loss_list.append(loss_val.item())
        schedular.step()
        print('[Epoch {}]\t Learning rate: {:.6f} | loss: {:.6f}'.format(epoch, Decimal(l_rate), np.mean(loss_list)))
        writer.add_scalar('train_pane/lr', l_rate, epoch)
        writer.add_scalar('train_pane/loss', np.mean(loss_list), epoch)
        #Evaluation
        if args.training.eval_when_train and epoch % args.ckp.save_by_epoch==0:
            time_start = torch.cuda.Event(enable_timing=True)
            time_end = torch.cuda.Event(enable_timing=True)
            time_list = []
            SRmodel.eval()
            psnr = []
            bar_eval = tqdm.tqdm(eval_loader)
            with torch.no_grad():
                for idx, (hr, lr, _) in enumerate(bar_eval):
                    hr = hr.to(device)
                    lr = lr.to(device)
                    time_start.record()
                    sr = SRmodel(lr)
                    time_end.record()
                    torch.cuda.synchronize()
                    time_list.append(time_start.elapsed_time(time_end))
                    sr = cutils.quantize(sr, args.training.rgb_range)

                    eval_psnr = cutils.calc_psnr(sr, hr, 4, args.training.rgb_range, benchmark=True)
                    psnr.append(eval_psnr)
                ave_runtime = sum(time_list) / len(time_list) / 1000.0
                print('[Epoch {}] ===> PSNR: {:.4f} | Average time: {:.6f} seconds'.format(epoch, np.mean(psnr), ave_runtime))
                writer.add_scalar('train_pane/psnr', np.mean(psnr), epoch)

        print('Save checkpoint')
        if not os.path.exists(os.path.join(args.ckp.save_root, cfg.config)):
            os.makedirs(os.path.join(args.ckp.save_root, cfg.config))
        saveckp_path = os.path.join(args.ckp.save_root, cfg.config, args.ckp.ckp_path+'{}.pt'.format(epoch))
        torch.save(SRmodel.module.model.state_dict(), saveckp_path)


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
    ''' Training
    '''
    net_train(args, cfg)

