import torch
from torch.utils.data import Dataset
from os import listdir
from glob import glob
import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from utils.misc import gridRows
from scipy import ndimage as ndi
import pandas as pd

class PatientImage:
    def __init__(self, img_path, data_device="cpu"):
        self.img_path = img_path
        
        self.dev = torch.device(data_device)

    def imgObject(self):
        img = nib.load(self.img_path)
        
        return img
    
    def size(self):
        img = self.imgObject()
        Nx,Ny,_ = img.shape
        
        return Nx,Ny
    
    def dimensions(self):
        img = self.imgObject()
        hdr = img.header
        zm = hdr.get_zooms()
        
        dx = zm[0]
        dy = zm[1]
        
        return dx,dy
    
    def coordinates(self):
        Nx,Ny = self.size()
        
        img = self.imgObject()
        
        i = np.arange(0,Nx);
        j = np.arange(0,Ny);
        k = np.arange(0,1);
        
        ii,jj,kk = np.meshgrid(i,j,k)
        ii = ii.reshape((Nx*Ny,1))
        jj = jj.reshape((Nx*Ny,1))
        kk = kk.reshape((Nx*Ny,1))
        II = np.concatenate((ii,jj,kk),axis=1)
        XX = apply_affine(img.affine,II)
        
        x = np.squeeze(XX[:,0].reshape((Nx,Ny)))
        y = np.squeeze(XX[:,1].reshape((Nx,Ny)))
        
        x = x[0,:].flatten()
        y = y[:,0].flatten()
    
        return x,y
    
    def extent(self):
        Nx,Ny = self.size()
        dx,dy = self.dimensions()
        
        Dx = dx*Nx
        Dy = dy*Ny
        
        return Dx,Dy
    
    def imageCenter(self):
        x,y = self.coordinates()
        xc = np.mean(x)
        yc = np.mean(y)
        
        return xc,yc
        
    def imageData(self):
        img = self.imgObject()
        I = img.get_fdata()
        I = np.transpose(np.squeeze(I[:,:,0]))
        
        return I
    
    def imageTensor(self):        
        I = self.imageData()        
        Nx,Ny = self.size()
        
        It = torch.zeros(1,Nx,Ny, device = self.dev)        
        It[0,:,:] = torch.from_numpy(imgdata)
        
        return It
    
    
    def crop(self,xc,yc,D,N):
        I = self.imageData()
        dx,dy = self.dimensions()
        x0,y0 = self.imageCenter()
        
        mci = InterpolatedMultichannelImage(I,dx=dx,dy=dy,x0=x0,y0=y0)
        Ic = mci.crop(xc,yc,D,N,cval=-3024,circ=True)
        
        return np.squeeze(Ic)
    
    def cropTensor(self,xc,yc,D,N):
        Ic = self.crop(xc,yc,D,N)
        Nx,Ny = self.size()
        Ict = torch.zeros(1,Nx,Ny,device = self.dev)
        Ict[0,:,:] = torch.from_numpy(Ic)
        
        return Ict

class PatientSegmentation:
    def __init__(self, seg_path, data_device="cpu"):
        self.seg_path = seg_path
        
        self.dev = torch.device(data_device)
        
    def classes(self):
        segdata = self.indexedData()
        cs = np.unique(segdata)
        
        return cs

    def segObject(self):
        seg = nib.load(self.seg_path)
        
        return seg
    
    def size(self):
        seg = self.segObject()
        Nx,Ny,_ = seg.shape
        
        return Nx,Ny
    
    def dimensions(self):
        seg = self.segObject()
        hdr = seg.header
        zm = hdr.get_zooms()
        
        dx = zm[0]
        dy = zm[1]
        
        return dx,dy
    
    def coordinates(self):
        Nx,Ny = self.size()
        
        seg = self.segObject()
        
        i = np.arange(0,Nx);
        j = np.arange(0,Ny);
        k = np.arange(0,1);
        
        ii,jj,kk = np.meshgrid(i,j,k)
        ii = ii.reshape((Nx*Ny,1))
        jj = jj.reshape((Nx*Ny,1))
        kk = kk.reshape((Nx*Ny,1))
        II = np.concatenate((ii,jj,kk),axis=1)
        XX = apply_affine(seg.affine,II)
        
        x = np.squeeze(XX[:,0].reshape((Nx,Ny)))
        y = np.squeeze(XX[:,1].reshape((Nx,Ny)))
        
        x = x[0,:].flatten()
        y = y[:,0].flatten()
    
        return x,y
    
    def extent(self):
        Nx,Ny = self.size()
        dx,dy = self.dimensions()
        
        Dx = dx*Nx
        Dy = dy*Ny
        
        return Dx,Dy
    
    def segmentationCenter(self):
        x,y = self.coordinates()
        xc = np.mean(x)
        yc = np.mean(y)
        
        return xc,yc
        
    def indexedData(self):
        seg = self.segObject()
        S = seg.get_fdata()
        S = np.transpose(np.squeeze(S[:,:,0]))

#         Nx,Ny = self.size()
#         N = np.maximum(Nx,Ny)
#         ii = np.arange(0,N)
#         ii,jj = np.meshgrid(ii,ii)
#         M = 1.0*((ii-N/2)**2+(jj-N/2)**2 < (N/2)**2)  
        return S
    
    def indexedTensor(self):        
        S = self.indexedData()        
        Nx,Ny = self.size()
        
        St = torch.from_numpy(imgdata)
        
        return St
    
    def multicomponentData(self):
        S = self.indexedData()
        Nx,Ny = self.size()
        cs = self.classes()
        Nc = len(cs)
        
        Sm = np.zeros((Nx,Ny,Nc))
        
        for ind,cls in enumerate(cs):
            Sm[:,:,ind] = 1.0*(S == cls)
            
        return Sm

    def crop(self,xc,yc,D,N):
        Sc = self.multicomponentData()
        dx,dy = self.dimensions()
        x0,y0 = self.segmentationCenter()
        
        mcs = InterpolatedMultichannelImage(Sc,dx=dx,dy=dy,x0=x0,y0=y0)
        Scc = mcs.crop(xc,yc,D,N)
        
        Scc = np.argmax(Scc,axis=2)+1
        ii = np.arange(0,N)
        ii,jj = np.meshgrid(ii,ii)
        M = 1.0*((ii-N/2)**2+(jj-N/2)**2 < (N/2)**2)        
        Scc = Scc*M
        
        return Scc
    
    def cropTensor(self,xc,yc,D,N):
        Sc = self.crop(xc,yc,D,N)
        Nx,Ny = self.size()
        Sct = torch.zeros(Nx,Ny,device = self.dev)
        Sct[:,:] = torch.from_numpy(Sc)
        
        return Sct

class InterpolatedMultichannelImage:
    def __init__(self, c,dx=1,dy=1,x0=0,y0=0):
        self.dx = dx
        self.dy = dy
        self.x0 = x0
        self.y0 = y0
        
        if c.ndim == 3:
            self.c = c
        elif c.ndim == 2:
            shp = c.shape
            self.c = np.zeros((shp[0],shp[1],1))
            self.c[:,:,0] = c
        else:
            raise ValueError('c must be 2 or 3 dimensional')
        
    def size(self):
        shp = self.c.shape
        Nx = shp[0]
        Ny = shp[1]
        
        return Nx,Ny
    
    def numChannels(self):
        shp = self.c.shape
        Nc = shp[2]
            
        return Nc
    
    def dimensions(self):
        dx = self.dx
        dy = self.dy
        
        return dx,dy
    
    def extent(self):
        Nx,Ny = self.size()
        dx,dy = self.dimensions()
        
        Dx = dx*Nx
        Dy = dy*Ny
        
    def center(self):
        x0 = self.x0
        y0 = self.y0
        
        return x0,y0
    
    def crop(self,xc,yc,D,N,cval=0,circ=False):
        dx,dy = self.dimensions()
        x0,y0 = self.center()
        Nx,Ny = self.size()
        Nc = self.numChannels()
        
#         print("dx: " + str(dx))
#         print("dy: " + str(dy))
#         print("x0: " + str(x0))
#         print("y0: " + str(y0))
#         print("Nx: " + str(Nx))
#         print("Ny: " + str(Ny))
        
        d = D/N
        
        sx = d/dx
        sy = d/dy
        fx = (xc-x0)/dx/sx
        fy = (yc-y0)/dy/sy
        
#         print("sx: " + str(sx))
#         print("sy: " + str(sy))
#         print("fx: " + str(fx))
#         print("fy: " + str(fy))
        
        M = [[sx,0,0],[0,sy,0],[0,0,1]]    
        os = (np.round(Nx/sx).astype(int),np.round(Ny/sy).astype(int),Nc)
        
#         print("os: " + str(os))
        
        Cc = ndi.affine_transform(self.c,M,output_shape=os,cval=cval)
        
        ix = np.round(np.arange(0,N)-N/2+os[0]/2+fx).astype(int)
        iy = np.round(np.arange(0,N)-N/2+os[1]/2-fy).astype(int)
        
#         print(ix)
#         print(iy)
        
        ix,iy = np.meshgrid(ix,iy)
        
        Cc = Cc[iy,ix,:]
        
        if circ:
            ii = np.arange(0,N)
            ii,jj,kk = np.meshgrid(ii,ii,0)
            M = 1.0*((ii-N/2)**2+(jj-N/2)**2 < (N/2)**2)
            Cc = Cc*M+(1-M)*cval
            
        return Cc

class RegressorDataset(Dataset):
    def __init__(self, imgs_dirs, segs_dirs, origin_path, data_device="cpu", D = None, N = None, xc=None, yc=None, centermode = None, mu = 0, sigma= 1):
        
        Ndirs = len(imgs_dirs)
        img_paths = []
        ids = []
        for dirInd in range(Ndirs):
            img_paths.extend([imgs_dirs[dirInd] + "/" + file for file in listdir(imgs_dirs[dirInd]) if file.endswith('.nii.gz')])
            ids.extend([file.replace('.nii.gz','') for file in listdir(imgs_dirs[dirInd])
                    if file.endswith('.nii.gz')])
        self.img_paths = img_paths
        self.ids = ids
        
        Ndirs = len(segs_dirs)
        segs = []
        for dirInd in range(Ndirs):
            segs.extend([segs_dirs[dirInd] + "/" + file for file in listdir(segs_dirs[dirInd]) if file.endswith('.nii.gz')])
        self.seg_paths = segs
                
        self.dev = torch.device(data_device)
        
        self.D  = D
        self.N  = N
        
        self.mode = centermode
        self.xc = xc
        self.yc = yc
        self.mu = mu
        self.sg = sigma
        
        org_csv = pd.read_csv(origin_path)
        
        self.bbdata = org_csv

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        
#         idx = self.ids[i]
#         img_file = glob(self.imgs_dir + idx + '.nii.gz')
#         seg_file = glob(self.segs_dir + idx + '.nii.gz')

        img_file = self.img_paths[i]
        seg_file = self.seg_paths[i]
        
        img = PatientImage(img_file)
        seg = PatientSegmentation(seg_file)
        
        if self.mode == None:
            xc,yc = img.imageCenter()
        elif self.mode == 'fixed':
            xc = self.xc
            yc = self.yc
        elif self.mode == 'rand':
            xc = self.mu[0] = self.sg*(2*np.random.rand()-1)
            yc = self.mu[1] = self.sg*(2*np.random.rand()-1)
        elif self.mode == 'randn':
            xc = self.mu[0] + self.sg*np.random.randn()
            yc = self.mu[1] + self.sg*np.random.randn()
        else:
            raise ValueError('Invalid mode')
            
        if self.D == None:
            Dx,Dy = img.extent()
            D = np.minimum(Dx,Dy)
        elif self.D <= 0:
            raise ValueError('D must be positive')
        else:
            D = self.D
        
        if self.N == None:
            Nx,Ny = img.size()
            N = np.minimum(Nx,Ny)
        elif self.N <= 0:
            raise ValueError('N must be positive')
        else:
            N = self.N
            
        I = img.cropTensor(xc,yc,D,N)
        S = seg.cropTensor(xc,yc,D,N)
        
        if self.dev == 'cuda':
            S = S.type(torch.cuda.LongTensor)
        else:
            S = S.type(torch.LongTensor)
        
        bx,by = self.getBodyOrigin(i)
        B = torch.zeros(2)
        B[0] = bx-xc
        B[1] = by-yc

        return {
            'img': I,
            'seg': S,
            'org': B
        }
    
    def getBodyOrigin(self,i):
        idx = self.ids[i].split("-")[0]
        match = self.bbdata['PatientName'].str.match(idx)
        bbrow = self.bbdata[match]
        bx = bbrow['OriginX2'].to_numpy()[0]
        by = bbrow['OriginY2'].to_numpy()[0]
        
        return bx,by
    
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