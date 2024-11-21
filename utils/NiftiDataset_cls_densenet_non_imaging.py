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
from torch.utils.data import Dataset

from monai.transforms import LoadImage, apply_transform

class NifitDataSet(torch.utils.data.Dataset):
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
                 split=3, num_ctrl=None):

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
        # filter out NAN based on label at month 24, after filtering, the rows go from 3217 to 2635 rows for training set
        # and 804 to 648 rows for testing set
        df = df[df['DX'].notna()]  # df['DX'].notna() returns True/False for each row
        # another round of filtering, this time we get 1460 rows for training set, and the same 648 rows for testing
        if label_time == 'bl':
            df = df[df['DX_bl'].notna()]
            self.df = df
            self.df = self.df[self.df['VISCODE'] == label_time].reset_index(drop=True)
        else:
            df = df[df['DX'].notna()]
            self.df = df
            self.df = self.df[(self.df['VISCODE']==label_time)|(self.df['VISCODE']=='bl')].reset_index(drop=True)
        if label_time == 'bl':
            self.label_map = {'CN': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3}
        else:
            self.label_map = {'CN':0, 'MCI':1, 'Dementia':2}

        self.MRIList = []
        self.PETList = []
        self.DemoList = []
        self.LabelList = []
        self.NonImageList = []
        self.subject_ids = []
        self.save_nonimg_info = []
        self.phase = phase
        self.num_ctrl = num_ctrl # control the number of subjects in each class
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
        

    def read_feasible_image(self, img_time='bl', labeltime='m24', control=None):
        '''
        Store the imaging data path into a list, and non-imaging data features into a list. When call forward, we can just get the image directly according to these lists.
        '''
        self.df = self.df.fillna(-1)
        self.df = self.df.groupby(['PTID']) # group the subjects according to their ID: PTID
        
        time_map = {'bl':'Month0', 'm03':'Month3', 'm06':'Month6','m12':'Year1','m24':'Year2'}
        label_map = self.label_map
        # filter the patient id with both MRI and PET data
        count = 0
        cnt_group = 0
        for name, group in self.df:
            # total group number for training is 892, for testing is 218
            cnt_group += 1
            # print(group)
            # group may contain two or even more rows, but with the same subject ID
            group = group.reset_index(drop=True) # reset the index of group to 0,1,...
            subject = group.iloc[0].loc['PTID'] # get the subject ID
            
            if labeltime == 'bl':
                label = group.loc[group['VISCODE'] == labeltime].iloc[0].loc['DX_bl']
            else:
                if group.loc[group['VISCODE'] == labeltime].empty:
                    # 319 empty for training, 79 empty for testing
                    count += 1
                    continue
                else:
                    # img_time is always baseline ('bl')
                    if not time_map[img_time] in self.df_data_path[subject]['MRI']:
                        continue
                    if not time_map[img_time] in self.df_data_path[subject]['PET']:
                        continue
                    label = group.loc[group['VISCODE'] == labeltime].iloc[0].loc['DX']
                    label_bl = group.loc[group['VISCODE'] == labeltime].iloc[0].loc['DX_bl']
            if label not in label_map:
                continue # this if statement filters out nothing
            # For training set, after the above filtering
            # Length of label list: 297, non-image list: 297, MRI/PET list: 297
            # Image number for class 0-2: 126, 122, 49
            # For testing set, length of label list: 82, non-image list: 82, MRI/PET list: 82
            # Image number for class 0-2: 38, 30, 14

            # the input arg control is 'MCI' in the readme.md
            if control is not None:
                # 'MC' not in label_bl, only 'MC' in the label of baseline visit, this case is added to training set
                if control[0:2] not in label_bl:
                    continue

            # store the information
            if self.num_ctrl is None:
                self.LabelList.append(label_map[label])
                self.MRIList.append(self.df_data_path[subject]['MRI'][time_map[img_time]][0])
                self.PETList.append(self.df_data_path[subject]['PET'][time_map[img_time]][0])
                self.subject_ids.append(subject)
            else:
                if self.LabelList.count(label_map[label]) < self.num_ctrl:
                    self.LabelList.append(label_map[label])
                    self.MRIList.append(self.df_data_path[subject]['MRI'][time_map[img_time]][0])
                    self.PETList.append(self.df_data_path[subject]['PET'][time_map[img_time]][0])
                    self.subject_ids.append(subject)
                else:
                    continue

            if group.loc[group['VISCODE'] == img_time].empty:
                self.NonImageList.append(self.read_non_image(group.loc[group['VISCODE'] == labeltime].iloc[0]))
                nonimg_info = group.loc[group['VISCODE'] == labeltime].iloc[0]
                nonimg_info.loc['DX'] = group.loc[group['VISCODE'] == labeltime]['DX']
                self.save_nonimg_info.append(nonimg_info)
            else:
                self.NonImageList.append(self.read_non_image(group.loc[group['VISCODE'] == img_time].iloc[0]))
                nonimg_info = group.loc[group['VISCODE'] == img_time].iloc[0]
                nonimg_info.loc['DX'] = group.loc[group['VISCODE'] == labeltime]['DX'].iloc[0]
                self.save_nonimg_info.append(nonimg_info)

        print('Length of label list: {}, non-image list: {}, MRI/PET list: {}'.format(
            len(self.LabelList), len(self.NonImageList), len(self.MRIList)))
        print('Image number for class 0-2: {}, {}, {}'.format(
            self.LabelList.count(0), self.LabelList.count(1), self.LabelList.count(2)))
        # for training set, these two prints 200 200 200; and 44 118 38 respectively
        # for testing set, 48 48 48; 14 28 6
        # the following one print: total group number 892, empty number 319
        print('total group number {}, empty number {}'.format(cnt_group, count))
        df = pd.DataFrame(self.subject_ids)
        if not os.path.exists('labeled_{}_subject_ids.csv'.format(self.phase)):
            df.to_csv('labeled_{}_subject_ids.csv'.format(self.phase), index=False)
        df_nonimg = pd.DataFrame(self.save_nonimg_info)
        if not os.path.exists('labeled_{}_nonimg_info.csv'.format(self.phase)):
            df_nonimg.to_csv('labeled_{}_nonimg_info.csv'.format(self.phase), index=False)

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
        image_MRI = self.loader(MRI_path)
        image_PET = self.loader(PET_path)
        non_image = np.array(self.NonImageList[index])

        MRI_np = image_MRI
        PET_np = image_PET
        if self.transforms is not None:
            MRI_np = apply_transform(self.transforms, MRI_np, map_items=False)
            PET_np = apply_transform(self.transforms, PET_np, map_items=False)
        assert not np.any(np.isnan(non_image))
        return MRI_np, PET_np, label, torch.FloatTensor(non_image[:])

    def __len__(self):
        return len(self.LabelList)

