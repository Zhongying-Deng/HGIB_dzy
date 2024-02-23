import os
import torch
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_dir(path):              # if folder does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)


def new_state_dict(file_name):
    state_dict = torch.load(file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def GetFeatures(loaders, model):
    # extract featrues from pre-trained model
    # stack them 
    MRI = None
    PET = None
    Non_Img = None
    Label = None
    length = [0, 0]
    for idx, loader in enumerate(loaders):
        for i, data in enumerate(tqdm(loader)):
            model.set_input(data)
            i_MRI, i_PET, i_Non_Img = model.ExtractFeatures()
            if MRI is None:
                MRI = i_MRI
                PET = i_PET
                Non_Img = i_Non_Img
                Label = data[2]
            else:
                MRI = torch.cat([MRI, i_MRI], 0)
                PET = torch.cat([PET, i_PET], 0)
                Non_Img = torch.cat([Non_Img, i_Non_Img], 0)
                Label = torch.cat([Label, data[2]], 0)
        length[idx] = MRI.size(0)
    length[1] = length[1] - length[0]
    return MRI.cpu().detach().numpy(), PET.cpu().detach().numpy(), Non_Img.cpu().detach().numpy(), Label, length

def cal_metrics(CM):
    FP = CM.sum(axis=0) - np.diag(CM)
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    # print(FP)
    FP = np.array(FP)
    FN = np.array(FN)
    TP = np.array(TP)
    TN = np.array(TN)
    PPV = TP / (TP + FP +1e-8)
    NPV = TN / (FN + TN+1e-8)

    return PPV, NPV
