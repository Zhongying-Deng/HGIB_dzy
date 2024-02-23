import SimpleITK as sitk
import os
import re
import numpy as np
import random
import glob
import scipy.ndimage.interpolation as interpolation
import scipy
import torch
import torch.utils.data
import pandas as pd
import pickle
import pdb
import copy

import torchio as tio
from torch.utils.data import Dataset

from monai.transforms import LoadImage, apply_transform

class NifitSemiSupDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 which_direction='AtoB',
                 transforms=None,
                 shuffle_labels=False,
                 train=False,
                 test=False, 
                 phase='train',
                 non_imaging_data = False, 
                 label_time='m24',
                 control = None,
                 root_path = "/rds/project/rds-LlrDsbHU5UM/sw991/Project_AD/data/ADNI-rawdata/",
                 split=3,
                 use_strong_aug=False,
                 transforms_strong=None
                 ):

        # Init membership variables
        self.data_path = data_path
        self.root = root_path
        # change it bafore running
        df = pd.read_csv("../../script_pre_processing/%s_info%d.csv"%(phase, split), low_memory=False)
        df_path_filename = "../../script_pre_processing/%s_split%d.pkl"%(phase, split)

        self.df_data_path = None
        with open(df_path_filename, 'rb') as file_:
            unpickler = pickle.Unpickler(file_)
            self.df_data_path = unpickler.load()
        self.df = df

        self.label_map = {'CN': 0, 'EMCI': 1, 'LMCI': 2, 'SMC':3, 'AD': 4}

        self.MRIList = []
        self.PETList = []
        self.DemoList = []
        self.LabelList = []
        self.NonImageList = []
        self.length = 0
        self.non_imaging_data = non_imaging_data
        self.non_image_features = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE', 'ADNI_MEM', 'ADNI_EF',]
        self.non_image_features_range = {'AGE':[55.0, 91.4,], 'PTGENDER':['Male', 'Female'], 'PTEDUCAT':[6.0, 20.0],
                                         'APOE4':[0.0,2.0], 'MMSE':[9.0,30.0], 'ADNI_MEM':[-2.863, 3.055], 'ADNI_EF':[-2.855,2.865]}


        self.gender_map = {'Male': 0, 'Female': 1}
        self.read_feasible_image(labeltime=label_time, control=control)

        self.which_direction = which_direction
        self.transforms = transforms

        self.shuffle_labels = shuffle_labels
        self.train = train
        self.test = test
        self.loader = LoadImage(None, True, np.float32)
        self.bit = sitk.sitkFloat32

        self.use_strong_aug = use_strong_aug
        self.transforms_strong = transforms_strong
        self.rand_gamma = tio.RandomGamma(log_gamma=(0.70, 1.50), p=0.3)
        

    def read_feasible_image(self, img_time='bl', labeltime='m24', control=None):
        '''
        Store the imaging data path into a list, and non-imaging data features into a list. When call forward, we can just get the image directly according to these lists.
        '''
        self.df = self.df.fillna(-1)
        self.df = self.df.groupby(['PTID'])
        
        time_map = {'bl':'Month0', 'm03':'Month3', 'm06':'Month6','m12':'Year1','m24':'Year2'}
        label_map = self.label_map
        # filter the patient id with both MRI and PET data
        count = 0
        cnt_group = 0
        for name, group in self.df:
            cnt_group += 1
            # print(group)
            # group may contain two or even more rows, but with the same subject ID
            group = group.reset_index(drop=True) # reset the index of group to 0,1,...
            subject = group.iloc[0].loc['PTID'] # get the subject ID
            
            if not group.loc[group['VISCODE'] == img_time].empty:
                    count += 1
                    if not time_map[img_time] in self.df_data_path[subject]['MRI']:
                        continue
                    if not time_map[img_time] in self.df_data_path[subject]['PET']:
                        continue
                    label = group.loc[group['VISCODE'] == img_time].iloc[0].loc['DX_bl']
            else:
                continue

            # store the information
            self.LabelList.append(label_map[label])
            self.MRIList.append(self.df_data_path[subject]['MRI'][time_map[img_time]][0])
            self.PETList.append(self.df_data_path[subject]['PET'][time_map[img_time]][0])
            self.NonImageList.append(self.read_non_image(group.loc[group['VISCODE'] == img_time].iloc[0]))

        print('Length of unlabeled list: {}, non-image list: {}, MRI/PET list: {}'.format(
            len(self.LabelList), len(self.NonImageList), len(self.MRIList)))
        print('Image number for class 0-4: {}, {}, {}, {}, {}'.format(
            self.LabelList.count(0), self.LabelList.count(1), self.LabelList.count(2),
            self.LabelList.count(3), self.LabelList.count(4)))
        print('total group number {}, non-empty number (having baseline visit) {} for unlabeled data'.format(
            cnt_group, count))

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def read_non_image(self, group):
        features = []
        for item in self.non_image_features:
            if item == "PTGENDER":
                features.append(self.gender_map[group.loc[item]])
            else:
                features.append(self.normalize_non_image(group.loc[item], item))
        return features

    def normalize_non_image(self, num, item):
        mmin = self.non_image_features_range[item][0]
        mmax = self.non_image_features_range[item][1]
        return ((num-mmin)/(mmax-mmin+1e-5))*2.0-1.0


    def __getitem__(self, index):
        label = self.LabelList[index]
        # read image and label
        MRI_path = os.path.join(self.root, self.MRIList[index])
        PET_path = os.path.join(self.root, self.PETList[index])
        image_MRI = self.loader(MRI_path) # shape ([121, 145, 61])
        image_PET = self.loader(PET_path)
        non_image = np.array(self.NonImageList[index])

        MRI_np = copy.deepcopy(image_MRI)
        PET_np = copy.deepcopy(image_PET)

        if self.transforms is not None:
            # torch.Size([1, 96, 96, 96])
            MRI_np = apply_transform(self.transforms, MRI_np, map_items=False)
            PET_np = apply_transform(self.transforms, PET_np, map_items=False)
        assert not np.any(np.isnan(non_image))

        if self.use_strong_aug:
            MRI_str_aug = image_MRI
            PET_str_aug = image_PET
            if self.transforms_strong is not None:
                MRI_str_aug = apply_transform(self.transforms_strong, MRI_str_aug, map_items=False)
                PET_str_aug = apply_transform(self.transforms_strong, PET_str_aug, map_items=False)
            else:
                MRI_str_aug = apply_transform(self.transforms, MRI_str_aug, map_items=False)
                PET_str_aug = apply_transform(self.transforms, PET_str_aug, map_items=False)
            return MRI_np, PET_np, label, torch.FloatTensor(non_image[:]), MRI_str_aug, PET_str_aug
        return MRI_np, PET_np, label, torch.FloatTensor(non_image[:])

    def __len__(self):
        return len(self.LabelList)

