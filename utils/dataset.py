from os import listdir
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import nibabel as nib
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, segs_dir, data_device="cpu"):
        self.imgs_dir = imgs_dir
        self.segs_dir = segs_dir
        
        self.ids = [file.replace('.nii.gz','') for file in listdir(imgs_dir)
                    if file.endswith('.nii.gz')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
        self.dev = torch.device(data_device)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.nii.gz')
        seg_file = glob(self.segs_dir + idx + '.nii.gz')
        
        img = nib.load(img_file[0])
        seg = nib.load(seg_file[0])
        
        Nx,Ny,_ = img.shape
        
        imgdata = img.get_fdata()
        imgdata = np.squeeze(imgdata)
        segdata = seg.get_fdata()
        segdata = np.squeeze(segdata)

        I = torch.zeros(1,Nx,Ny, device = self.dev)
        S = torch.zeros(Nx,Ny, device = self.dev)
        
        I[:,:,:] = torch.from_numpy(imgdata)
        S[:,:] = torch.from_numpy(segdata)
        
        if self.dev == 'cuda':
            S = S.type(torch.cuda.LongTensor)
        else:
            S = S.type(torch.LongTensor)
        
        
#         assert I.shape == S.shape, \
#             f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        return {
            'img': I,
            'seg': S
        }
    
    def __nclass__(self):
        item = self.__getitem__(0)
        
        seg = item["seg"].numpy()
        
        return np.max(seg)
    
    def __str__(self):
        
        printstr = "BasicDataset with " + str(len(self.ids)) + " items. "
        
        item = self.__getitem__(0)
        
        Nc,Nx,Ny = item["img"].shape
        
        printstr += "Image dim: " + str(Nx) + "x" + str(Ny) + ". Num channels: " + str(Nc) + ". "
        
        printstr += "Device: " + str(self.dev) + ". "
        
        return printstr

#     @classmethod
#     def preprocess(cls, pil_img, scale):
#         w, h = pil_img.size
#         newW, newH = int(scale * w), int(scale * h)
#         assert newW > 0 and newH > 0, 'Scale is too small'
#         pil_img = pil_img.resize((newW, newH))

#         img_nd = np.array(pil_img)

#         if len(img_nd.shape) == 2:
#             img_nd = np.expand_dims(img_nd, axis=2)

#         # HWC to CHW
#         img_trans = img_nd.transpose((2, 0, 1))
#         if img_trans.max() > 1:
#             img_trans = img_trans / 255

#         return img_trans