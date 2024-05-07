import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import torch.nn.functional as F
import math
import numpy as np

from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio
from pytorch_fid import fid_score
from dataset import plot_3d_from_tensor,plot_2d_from_tensor



def Psnr_tensor2d(tensor_hr,tensor_sr,slice_axis):

    tensor_hr = tensor_hr.squeeze().cpu().numpy()
    tensor_hr = tensor_hr
    h,w,d = tensor_hr.shape
    slice_index = int(w/2)
    if slice_axis == 0:
        slice_hr = tensor_hr[slice_index, :, :]
    elif slice_axis == 1:
        slice_hr = tensor_hr[:, slice_index, :]   
    elif slice_axis == 2:
        slice_hr = tensor_hr[:, :, slice_index]         

    tensor_hr = slice_hr

    tensor_sr = tensor_sr.squeeze().cpu().numpy()
    if len(tensor_sr.shape) >=3:
        # 选择切片轴
        if slice_axis == 0:
            
            slice_sr = tensor_sr[slice_index, :, :]
        elif slice_axis == 1:
            
            slice_sr = tensor_sr[:, slice_index, :]
        elif slice_axis == 2:
            
            slice_sr = tensor_sr[:, :, slice_index]      
        else:
            raise ValueError("Invalid slice_axis. Choose from 0, 1, or 2.")
        tensor_sr = slice_sr

    
    # 计算 PSNR
    psnr_value = peak_signal_noise_ratio(tensor_hr, tensor_sr, data_range=tensor_hr.max() - tensor_hr.min())
    
    return psnr_value
def Psnr_tensor(tensor_hr,tensor_sr):
    tensor_hr = tensor_hr.squeeze().cpu().numpy()
    tensor_hr = tensor_hr
    
    tensor_sr = tensor_sr.squeeze().cpu().numpy()
    tensor_sr = tensor_sr
    
    # 计算 PSNR
    psnr_value = peak_signal_noise_ratio(tensor_hr, tensor_sr, data_range=tensor_hr.max() - tensor_hr.min())
    
    return psnr_value


def Ssim_tensor(tensor_hr,tensor_sr):
    tensor_hr = tensor_hr.squeeze().cpu().detach().numpy()
    tensor_sr = tensor_sr.squeeze().cpu().detach().numpy()
    ssim_value = 0.0
    for i in range(tensor_hr.shape[2]):
        # 计算每个深度层的SSIM
        ssim_value += ssim_metric(tensor_hr[:, :, i], tensor_sr[:, :, i], data_range=tensor_hr.max() - tensor_hr.min())

    # 取平均值
    ssim_value /= tensor_hr.shape[2]

    return ssim_value


def MAE_tensor(tensor_hr,tensor_sr):
    # data1 和 data2 的维度为（64，64，128）

    # 将 torch.Tensor 转为 numpy.ndarray
    tensor_hr = tensor_hr.squeeze().cpu().numpy()
    tensor_sr = tensor_sr.squeeze().cpu().numpy()

    # 计算 FID
    mae = np.abs(tensor_hr - tensor_sr).mean()
    
    return mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, 
                        default="/mnt/HDD2/hlz/sr3_drnn/config/sr_ddpm_16_128.json",
                        #'/mnt/HDD2/hlz/SR3_/config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    avg_psnr = []
    avg_ssim = []
    avg_mae = []

    avg_inf_psnr = []
    avg_inf_ssim = []
    avg_inf_mae = []
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=True)

        # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
        hr_img = visuals['HR']
        lr_img = visuals["LR"]
        # plot_3d_from_tensor(lr_img, '{}/{}_{}_lr3d'.format(result_path, current_step, idx))
        # plot_2d_from_tensor(lr_img, '{}/{}_{}_lr2d'.format(result_path, current_step, idx),  slice_axis=1)
        sr_data = visuals["INF"]
        # plot_3d_from_tensor(sr_data, '{}/{}_{}_sr3d'.format(result_path, current_step, idx))
        # plot_2d_from_tensor(sr_data, '{}/{}_{}_sr2d'.format(result_path, current_step, idx),  slice_axis=1)
        sr_img_mode = 'single'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR'][-1] 
            psnr = Psnr_tensor(hr_img,sr_img)
            ssim = Ssim_tensor(hr_img,sr_img)
            mae = MAE_tensor(hr_img,sr_img)
           

            psnr_inf = Psnr_tensor(hr_img,sr_data)
            avg_inf_psnr.append(psnr_inf)

            ssim_inf = Ssim_tensor(hr_img,sr_data)
            avg_inf_ssim.append(ssim_inf)

            mae_inf = MAE_tensor(hr_img,sr_data)
            avg_inf_mae.append(mae_inf)

            print("psnr_inf:  "+str(psnr_inf))
            print("ssim_inf:  "+str(ssim_inf))
            print("mae_inf: "+str(mae_inf))
            print("psnr:  "+str(psnr))
            print("ssim:  "+str(ssim))
            print("mae:  "+str(mae))
            plot_3d_from_tensor(sr_data, '{}/{}_{}_inf3d'.format(result_path, current_step, idx))
            plot_2d_from_tensor(sr_data, '{}/{}_{}_inf2d'.format(result_path, current_step, idx),  slice_axis=1)

            plot_3d_from_tensor(lr_img, '{}/{}_{}_lr3d'.format(result_path, current_step, idx))
            plot_2d_from_tensor(lr_img, '{}/{}_{}_lr2d'.format(result_path, current_step, idx),  slice_axis=1)

            plot_3d_from_tensor(hr_img, '{}/{}_{}_hr3d'.format(result_path, current_step, idx))
            plot_2d_from_tensor(hr_img, '{}/{}_{}_hr2d'.format(result_path, current_step, idx),  slice_axis=1)

            plot_3d_from_tensor(sr_img, '{}/{}_{}_sr3d'.format(result_path, current_step, idx))
            plot_2d_from_tensor(sr_img, '{}/{}_{}_sr2d'.format(result_path, current_step, idx),  slice_axis=1)
            
        elif sr_img_mode == 'zhuo':
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                print(iter)
                plot_3d_from_tensor(sr_img[iter], '{}/{}_{}_sr_{}'.format(result_path, current_step, idx,iter))
                plot_2d_from_tensor(sr_img[iter], '{}/{}_{}_sr2d_{}'.format(result_path, current_step, idx,iter),  slice_axis=1)
        elif sr_img_mode == 'gride':
            # grid img
            # sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            sr_img = visuals['SR'][-1] 
            psnr = Psnr_tensor(hr_img,sr_img)
            print("psnr:"+str(psnr))
            # avg_psnr.append(psnr)
            lr_data = lr_img.squeeze()

            lr_data_bic = F.interpolate(lr_data[:,16,:].unsqueeze(0).unsqueeze(0), 
                                      size=(64,128), mode='bicubic', align_corners=False)
            lr_data_bil = F.interpolate(lr_data[:,16,:].unsqueeze(0).unsqueeze(0), 
                                      size=(64,128), mode='bilinear', align_corners=False)
            lr_data_nea = F.interpolate(lr_data[:,16,:].unsqueeze(0).unsqueeze(0), 
                                      size=(64,128), mode='nearest')
            

            # interpolated_data = F.interpolate(lr_data[:,16,:].unsqueeze(0), 
            #                           size=(64,128), mode='linear')
            # lr_data_lin = interpolated_data.squeeze().cpu().numpy()

            psnr_2d = Psnr_tensor2d(hr_img,sr_img,slice_axis=1)
            psnr_inf_2d = Psnr_tensor2d(hr_img,sr_data,slice_axis=1)
            psnr_bic_2d = Psnr_tensor2d(hr_img,lr_data_bic,slice_axis=1)
            psnr_bil_2d = Psnr_tensor2d(hr_img,lr_data_bil,slice_axis=1)
            psnr_nea_2d = Psnr_tensor2d(hr_img,lr_data_nea,slice_axis=1)
            # psnr_lin_2d = Psnr_tensor2d(hr_img,lr_data_lin,slice_axis=1)

            mae_2d = MAE_tensor(hr_img,sr_img)
            mae_inf = MAE_tensor(hr_img,sr_data)
            mae_bic = MAE_tensor(hr_img,lr_data_bic)
            mae_bil = MAE_tensor(hr_img,lr_data_bil)
            mae_nea = MAE_tensor(hr_img,lr_data_nea)

            print("psnr_2d: "+str(psnr_2d))
            print("mae_2d: "+str(mae_2d))

            print("psnr_inf_2d: "+str(psnr_inf_2d))
            print("mae_inf: "+str(mae_inf))

            print("psnr_bic_2d: "+str(psnr_bic_2d))
            print("mae_bic: "+str(mae_bic))

            print("psnr_bil_2d: "+str(psnr_bil_2d))
            print("mae_bil: "+str(mae_bil))

            print("psnr_nea_2d: "+str(psnr_nea_2d))
            print("mae_nea: "+str(mae_nea))

            # print("psnr_lin_2d: "+str(psnr_lin_2d))
            plot_2d_from_tensor(hr_img, '{}/{}_{}_hr2d'.format(result_path, current_step, idx),  slice_axis=1)
            plot_2d_from_tensor(sr_img, '{}/{}_{}_sr2d'.format(result_path, current_step, idx),  slice_axis=1)
            plot_2d_from_tensor(sr_data, '{}/{}_{}_sr_inf2d'.format(result_path, current_step, idx),  slice_axis=1)
            plot_2d_from_tensor(lr_data_bic, '{}/{}_{}_sr_bic2d'.format(result_path, current_step, idx),  slice_axis=1)
            plot_2d_from_tensor(lr_data_bil, '{}/{}_{}_sr_bil2d'.format(result_path, current_step, idx),  slice_axis=1)
            plot_2d_from_tensor(lr_data_nea, '{}/{}_{}_sr_nea2d'.format(result_path, current_step, idx),  slice_axis=1)
  
    # avgpsnr = sum(avg_psnr)/len(avg_psnr)
    # print("avg psnr:"+str(avgpsnr))
    # avgssim = sum(avg_ssim)/len(avg_ssim)
    # print("avg ssim:"+str(avgssim))
    # avgmse = sum(avg_mae)/len(avg_mae)
    # print("avg mse:"+str(avgmse))

    # avginfpsnr = sum(avg_inf_psnr)/len(avg_inf_psnr)
    # print("avg inf psnr:"+str(avginfpsnr))
    # avginfssim = sum(avg_inf_ssim)/len(avg_inf_ssim)
    # print("avg inf ssim:"+str(avginfssim))
    # avginfmse = sum(avg_inf_mae)/len(avg_inf_mae)
    # print("avg inf mse:"+str(avginfmse))

