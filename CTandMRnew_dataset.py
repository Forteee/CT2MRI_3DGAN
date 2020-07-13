"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
#from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
from torch.utils import data
import SimpleITK as sitk
from skimage.measure import label 
import cv2
#from PIL import Image
import os
import numpy as np


def setDicomWinWidthWinCenter(img_data, winwidth=100, wincenter=35, rows=512, cols=512):
    # window of brain tissue: -35 --- 85
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 
    max = (2 * wincenter + winwidth) / 2.0 
    dFactor = 255.0 / (max - min)
    #dFactor = 1.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)  # 相当于把处于-35 --- 85 的HU值归一化到0-255 

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255    
    return img_temp


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity = 1, background=0)
    
    if len(np.bincount(labels.flat)[1:]) ==0:
        return 0
    else:
        largestCC = np.zeros_like(labels)
        label_counts = np.bincount(labels.flat)[1:]
        
        # Keep the largest connected compunent anyway
        largestCC = (labels == np.argmax(label_counts) + 1)

    return largestCC


class CTandMRnewDataset(data.Dataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # get the image paths of your dataset;
        self.dir_CT = os.path.join('data/CT&MR/testCT')
        self.dir_MR = os.path.join('data/CT&MR/testMR')

        #self.CT_paths = sorted(make_dataset(self.dir_CT, opt.max_dataset_size)) # 返回list,list每一个元素是./datasets/CT&MR/trainCT里每一个图的路径
        #self.MR_paths = sorted(make_dataset(self.dir_MR, opt.max_dataset_size))
        self.CT_paths = []
        self.MR_paths = []
        for p in os.listdir(self.dir_CT):
            self.CT_paths.append(os.path.join(self.dir_CT, p))
            self.MR_paths.append(os.path.join(self.dir_MR, p))
            

        self.CT_size = len(self.CT_paths)  
        self.MR_size = len(self.MR_paths) 
        

    def __getitem__(self, index):
         
        ct_path = self.CT_paths[index % self.CT_size] 
        mr_path = self.MR_paths[index % self.MR_size]
        # get CT data
        ct_img = sitk.ReadImage(ct_path)
        #pix_ct = sitk.GetArrayFromImage(ct_img)[5:21]
        pix_ct = sitk.GetArrayFromImage(ct_img)[7:19]

        mask_np_ct = np.logical_and(pix_ct > -15, pix_ct < 85).astype(np.uint8)
        tt = np.zeros_like(pix_ct)
        for i in range(tt.shape[0]):
            tt[i] = getLargestCC(mask_np_ct[i])
        for i in range(pix_ct.shape[0]):
            pix_ct[i, :, :][tt[i] == False]=0
        min_inmask = pix_ct.min()
        max_immask = pix_ct.max()
        pix_ctNorm = (pix_ct - min_inmask) / (max_immask - min_inmask)
        for i in range(pix_ctNorm.shape[0]):
            pix_ctNorm[i, :, :][tt[i] == False]=0  # pix_ct中有小于0的值 这一步将mask外的值再设一遍0 保证所有mask外的值都为0

        #data_CT = np.zeros((1, 512, 512, 16))
        data_CT = np.zeros((1, 256, 256, 12))
        for i in range(12):
            #curr = pix_ctNorm.transpose((1, 2, 0))[:, :, i]
            #data_CT[0, :, :, i] = np.pad(curr, ((15, 16), (55, 56)), mode='edge')  # shape: (512, 512, 16)
            curr = pix_ctNorm.transpose((1, 2, 0))[:, :, i]
            curr = np.delete(curr, 0, 0) 
            curr = np.pad(curr, ((0, 0), (39, 40)), mode='edge')
            data_CT[0, :, :, i] = cv2.resize(curr, (256, 256))  # shape: (480, 480, 16)
        # get MR data
        mr_img = sitk.ReadImage(mr_path)
        #pix_mr = sitk.GetArrayFromImage(mr_img)[5:21]
        pix_mr = sitk.GetArrayFromImage(mr_img)[7:19]

        for i in range(pix_ct.shape[0]):
            pix_mr[i, :, :][tt[i] == False]=0
        min_inmask = pix_mr.min()
        max_immask = pix_mr.max()
        pix_mrNorm = (pix_mr - min_inmask) / (max_immask - min_inmask)

        for i in range(pix_mrNorm.shape[0]):
            pix_mrNorm[i, :, :][tt[i] == False]=0
            
        #data_MR = np.zeros((1, 512, 512, 16))
        data_MR = np.zeros((1, 256, 256, 12))
        for i in range(12):
            #curr = pix_mrNorm.transpose((1, 2, 0))[:, :, i]
            #data_MR[0, :, :, i] = np.pad(curr, ((15, 16), (55, 56)), mode='edge')  # shape: (512, 512, 16)
            curr = pix_mrNorm.transpose((1, 2, 0))[:, :, i]
            curr = np.delete(curr, 0, 0) 
            curr = np.pad(curr, ((0, 0), (39, 40)), mode='edge')
            data_MR[0, :, :, i] = cv2.resize(curr, (256, 256))  # shape: (480, 480, 16)
            
        return {'CT': data_CT, 'CT_path': ct_path, 'MR': data_MR, 'MR_path': mr_path}

    def __len__(self):
        """Return the total number of images."""
        return max(self.CT_size, self.MR_size)

