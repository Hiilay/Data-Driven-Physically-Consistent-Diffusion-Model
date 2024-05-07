from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F

 
def custom_transform_sr(datasr,l_resolution,l_deep,r_resolution,r_deep,LR=False):
    datasr = torch.Tensor(datasr).view(l_resolution,l_resolution,l_deep)
    # 将结果恢复为目标大小
    if LR == False:
        interpolated_data = F.interpolate(datasr.unsqueeze(0).unsqueeze(0), 
                                      size=(r_resolution,r_resolution,r_deep), mode='trilinear', align_corners=False)
        interpolated_data = interpolated_data.squeeze(0).squeeze(0)
        transform = transforms.Compose([transforms.Normalize(0.5,0.5)])
        datasr = interpolated_data.unsqueeze(0)
        datasr = transform(datasr)
        return datasr
    if LR == True:
        transform = transforms.Compose([transforms.Normalize(0.5,0.5)])
        datalr = datasr.unsqueeze(0)
        datalr = transform(datasr)
        return datalr


def custom_transform_hr(datahr,r_resolution,r_deep):

    datahr = torch.Tensor(datahr).view(r_resolution,r_resolution,r_deep)
    transform = transforms.Compose([transforms.Normalize(0.5,0.5)])
    datahr = datahr.unsqueeze(0)
    datahr = transform(datahr)

    return datahr



class LRHRDataset(Dataset):
    def __init__(self, datasrroot,datahrroot,datatype, l_resolution=32, r_resolution=64,l_deep=64,r_deep=128,split='train', data_len=-1, need_LR=False,dataroot=None):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.l_deep = l_deep
        self.r_deep = r_deep
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype =="dat":
            # dataset_sr = CustomTxtDataset_sr(datasrroot,transform=custom_transform_sr)
            self.txt_files = [f for f in os.listdir(datasrroot) if f.endswith('.dat')]
            self.datasr = []
            # 逐个读取sr_dat文件
            for txt_file in self.txt_files:
                file_path = os.path.join(datasrroot, txt_file)
                with open(file_path, 'r') as file:
                    lines = file.readlines()[16:] # 从第16行开始读取
                    data = [float(item) for line in lines for item in line.split()]
                    data = np.array(data)
                    data = (data-data.min())/(data.max()-data.min())
                    self.datasr.append(data)

            #逐个读取hr_dat文件
            self.datahr = []
            #dataset_hr = CustomTxtDataset_hr(datahrroot,transform=custom_transform_hr)
            self.txt_files = [f for f in os.listdir(datahrroot) if f.endswith('.dat')]
            for txt_file in self.txt_files:
                file_path = os.path.join(datahrroot , txt_file)
                with open(file_path, 'r') as file:
                    lines = file.readlines()[16:]  # 从第16行开始读取
                    data = [float(item) for line in lines for item in line.split()]
                    data = np.array(data)
                    data = (data-data.min())/(data.max()-data.min())
                    self.datahr.append(data)

            self.dataset_len = len(self.datahr)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        elif self.datatype == 'dat':
            data_sr = self.datasr[index]
            data_hr = self.datahr[index]

            data_transofomer_sr =  custom_transform_sr(data_sr,self.l_res,self.l_deep,self.r_res,self.r_deep)   
            data_hr =  custom_transform_hr(data_hr,self.r_res,self.r_deep)  
            if self.need_LR:
                data_lr = custom_transform_sr(data_sr,self.l_res,self.l_deep,self.r_res,self.r_deep,LR=True) 
               
                return {"LR":data_lr,"HR":data_hr,"SR":data_transofomer_sr,"Index":index} 
            else:
                return {"HR":data_hr,"SR":data_transofomer_sr,"Index":index} 

            
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}
