from os import listdir
from glob import glob
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import logging
import nibabel as nib
import numpy as np
import math
from nibabel.affines import apply_affine
import scipy.ndimage.measurements as meas
from scipy import ndimage

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
        
class PatientSegmentation:
    def __init__(self, seg_file, data_device="cpu"):
        self.seg_file = seg_file
        
        self.dev = torch.device(data_device)

    def __size__(self):
        img = nib.load(self.seg_file)
        
        Nx,Ny,Nz = img.shape
        
        return Nx,Ny,Nz,img
    
    def coordinates(self,units="mm"):
        Nx,Ny,Nz,img = self.__size__()
        i = np.arange(0,Nx);
        j = np.arange(0,Ny);
        k = np.arange(0,Nz);
        
        ii,jj,kk = np.meshgrid(i,j,k)
        ii = ii.reshape((Nx*Ny*Nz,1))
        jj = jj.reshape((Nx*Ny*Nz,1))
        kk = kk.reshape((Nx*Ny*Nz,1))
        II = np.concatenate((ii,jj,kk),axis=1)
        
        XX = apply_affine(img.affine,II)
        
        x = XX[:,0].reshape((Nx,Ny,Nz))
        y = XX[:,1].reshape((Nx,Ny,Nz))
        z = XX[:,2].reshape((Nx,Ny,Nz))
    
        return x,y,z
   
    
    def interpolateOnCoordinates(self,i,j,k):
        Nx,Ny,Nz,img = self.__size__()
        ii = np.arange(0,Nx);
        jj = np.arange(0,Ny);
        kk = np.arange(0,Nz);
        
        xx,yy,zz = self.coordinates()
        xx = np.squeeze(xx[0,:,0])
        yy = np.squeeze(yy[:,0,0])
        zz = np.squeeze(zz[0,0,:])
        
        x = np.interp(i,ii,xx)
        y = np.interp(j,jj,yy)
        z = np.interp(k,kk,zz)
        
        return x,y,z
    
    def labelData(self):
        seg = nib.load(self.seg_file)
        segdata = seg.get_fdata()
        segdata = segdata.astype(int)
        lbls = np.unique(segdata)
        return segdata, lbls
        
    def centerOfMass(self,label):
        
        v,lbls = self.labelData()
        
        L = (v == label)*1.0
        com = meas.center_of_mass(L)
        comx = self.interpolateOnCoordinates(com[0],com[1],com[2])
        return com, comx
    
    def extent(self,label,prc_shift=5):
        
        v,_ = self.labelData()
        
        L = (v == label)*1.0
        
        xx,yy,zz = self.coordinates()
#         xmin = np.amin(xx[L==1])
#         xmax = np.amax(xx[L==1])
#         ymin = np.amin(yy[L==1])
#         ymax = np.amax(yy[L==1])
#         zmin = np.amin(zz[L==1])
#         zmax = np.amax(zz[L==1])
        
        xmin = np.percentile(xx[L==1], prc_shift)
        xmax = np.percentile(xx[L==1], 100-prc_shift)
        ymin = np.percentile(yy[L==1], prc_shift)
        ymax = np.percentile(yy[L==1], 100-prc_shift)
        zmin = np.percentile(zz[L==1], prc_shift)
        zmax = np.percentile(zz[L==1], 100-prc_shift)
        return xmin,xmax,ymin,ymax,zmin,zmax