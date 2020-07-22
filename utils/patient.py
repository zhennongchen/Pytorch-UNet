from os import listdir
from glob import glob
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import logging
import nibabel as nib
import numpy as np
import math

class Patient:
    def __init__(self, pat_file, data_device="cpu"):
        self.pat_file = pat_file
        
        self.dev = torch.device(data_device)

    def __size__(self):
        img = nib.load(self.pat_file)
        
        Nx,Ny,Nz = img.shape
        
        return Nx,Ny,Nz
    
    
    def predict(self,net,seg_file,batch_size=1):
        
        net.eval()
        
        img = nib.load(self.pat_file)

        Nx,Ny,Nz = img.shape
        
        imgdata = img.get_fdata()
        imgdata = np.squeeze(imgdata)
        imgdata = imgdata.transpose(2,0,1)
        
        Nbatch = math.ceil(Nz/batch_size)
        
        
        segdata = np.zeros((Nz,Nx,Ny))
        
        for batchInd in range(Nbatch):
            if batchInd < Nbatch-1:
                rng = batchInd*batch_size+np.arange(batch_size)
                I = torch.zeros(batch_size,1,Nx,Ny, device = self.dev)
                print(str(rng))
            else:
                rmndr = Nz%batch_size
                rng = batchInd*batch_size+np.arange(rmndr)
                I = torch.zeros(rmndr,1,Nx,Ny, device = self.dev)
                print(str(rng))
            I[:,0,:,:] = torch.from_numpy(imgdata[rng,:,:])
            I.to(device=self.dev)
            with torch.no_grad():
                P = net(I)
            P = torch.softmax(P,dim=1)
            _,S = torch.max(P,dim=1)
            S = S.to(device='cpu')
            segdata[rng,:,:] = S.numpy()
            
        segdata = segdata.transpose(1,2,0)
        
        seg = nib.Nifti1Image(segdata, img.affine)
        nib.save(seg, seg_file)