import sys
from os import listdir
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import nibabel as nib
import numpy as np

class CrossValDataset(Dataset):
    def __init__(self, data_dir, partition, testFlag = False, data_device="cpu"):
        print("Indexing data directory: " + data_dir + "...")
        
        self.path = data_dir
        
        self.groups = [directory for directory in listdir(data_dir)
                    if directory.startswith('group') ]
        
        self.partition = partition
        
        numGroups = len(self.groups)
        print(str(numGroups) + " groups found.")
        
        if partition < 0 or partition > (numGroups-1):
            raise ValueError("Partition outside range.")
        
        print("Partition selected: " + str(partition))
        
        self.dev = data_device
        
        self.testFlag = testFlag
        
        if testFlag:
            print("Serving test data.")
        else:
            print("Serving training data.")
            
    def getGroupItems(self,group):
        platform = sys.platform
        if platform == 'win32':
            pathsep = "\\"
        else:
            pathsep = "/"
        
        imgs = [file.replace('.nii.gz','') for file in listdir(self.path + pathsep + self.groups[group] + pathsep + "img" + pathsep)
                    if file.endswith('.nii.gz')]
        
        segs = [file.replace('.nii.gz','') for file in listdir(self.path + pathsep + self.groups[group] + pathsep + "seg" + pathsep)
                    if file.endswith('.nii.gz')]
        
        return {
            'img': imgs,
            'seg': segs
        }
    
    def getGroupLengths(self):
        numGroups = len(self.groups)
        
        groupLength = np.zeros((numGroups,1))
        for groupInd in range(numGroups):
            items = self.getGroupItems(groupInd)
            groupLength[groupInd] = len(items['img'])
            
        return groupLength
    
    def globalToGroupIndex(self,i):
        
        numGroups = len(self.groups)
        groupInds = range(numGroups)
        groupInds = [x for ii,x in enumerate(groupInds) if ( ii!=self.partition if not self.testFlag else ii==self.partition)]
        groupLength = self.getGroupLengths()
        groupLength = [x for ii,x in enumerate(groupLength) if ( ii!=self.partition if not self.testFlag else ii==self.partition)]
        groupMatches = [x for ii,x in enumerate(range(numGroups)) if( ii!=self.partition if not self.testFlag else ii==self.partition)]
        
        for groupInd in range(len(groupInds)):
            numPrevItems = np.sum(groupLength[0:groupInd])
            numItems = np.sum(groupLength[0:(groupInd+1)])
#             print("numPrev: " + str(numPrevItems) + ", num: " + str(numItems))
            if i > (numItems-1):
                continue
            else:
                groupMatch = groupMatches[groupInd]
                groupIndex = i-numPrevItems.astype(int)
                break
        
        return groupMatch,groupIndex
                
    
#     def groupLengths(self):
#         for groupInd in range(len(self.groups)):

    def __len__(self):
        groupLength = self.getGroupLengths()
        groupLength = [x for ii,x in enumerate(groupLength) if ( ii!=self.partition if not self.testFlag else ii==self.partition)]
        return np.sum(groupLength).astype(int)
            
        
    def __getitem__(self, i):
        platform = sys.platform
        if platform == 'win32':
            pathsep = "\\"
        else:
            pathsep = "/"
            
        groupMatch,groupIndex = self.globalToGroupIndex(i)
        
        items = self.getGroupItems(groupMatch)
        
        img_file = self.path + pathsep + self.groups[groupMatch] + pathsep + "img" + pathsep + items['img'][groupIndex] + ".nii.gz"
        seg_file = self.path + pathsep + self.groups[groupMatch] + pathsep + "seg" + pathsep + items['seg'][groupIndex] + ".nii.gz"
        
        img = nib.load(img_file)
        seg = nib.load(seg_file)
        
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
