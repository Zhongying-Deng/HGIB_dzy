import torch
import os
import numpy as np
from options.train_options import TrainOptions
import time
from models import create_model
from utils.visualizer import Visualizer
from tqdm import tqdm
from monai.data import DataLoader
from torchmetrics import ConfusionMatrix, Accuracy
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassSpecificity, MulticlassRecall, MulticlassAUROC
from utils.NiftiDataset_cls_densenet_non_imaging import NifitDataSet
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    CenterSpatialCrop,
)
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.utils import GetFeatures, cal_metrics

import random
random.seed(124)
np.random.seed(124)
torch.manual_seed(124)


def validation(model, epoch, writer):
    visualizer = Visualizer(opt)
    total_steps = 0

    losses = []
    acc = []

    visualizer.reset()
    total_steps += opt.batch_size

    model.validation()
    loss= model.get_current_losses(train=False)
    losses.append(loss)
    acc.append(model.get_current_acc())

    prediction = model.get_prediction_cur()
    target = model.get_target_cur()

    prediction = prediction.cpu().detach()

    target = target.cpu().detach()
    metric = MulticlassAccuracy(num_classes=num_classes, average='micro', thresholds=None)
    acc = metric(prediction, target)

    metric = MulticlassF1Score(num_classes=num_classes, average=None)
    F1 = metric(prediction, target)
    metric = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None)
    AUROC = metric(prediction, target)

    metric = MulticlassSpecificity(num_classes=num_classes, average=None)
    Specificity = metric(prediction, target)
    metric = MulticlassRecall(num_classes=num_classes, average=None)
    recall = metric(prediction, target)

    confmat = ConfusionMatrix(num_classes=num_classes)
    CM = confmat(prediction, target)

    PPV, NPV = cal_metrics(CM)

    print('[Validation] ', ' Accuracy:', acc, 'AUC', AUROC, 'PPV:', PPV, 'NPV:', NPV)


if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard/logs'))
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    # load dataset
    trainTransforms = Compose([ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    train_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True, phase='train', label_time=opt.label_time, control = opt.control, split=opt.split)
    print('length train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers*len(opt.gpu_ids), pin_memory=True)  # Here are then fed to the network with a defined batch size

    testTransforms = Compose([ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    test_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=testTransforms, shuffle_labels=False,
                           test=True, phase='test', label_time=opt.label_time, control=opt.control, split=opt.split)
    print('length test list:', len(test_set))
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True)

    # create model
    print("Creating model...")
    model = create_model(opt)  # creation of the model
    model.setup(opt)      

    model.load_pretrained_networks('latest')

    visualizer = Visualizer(opt)
    total_steps = 0
    # label time =bl [NC, EMCI, LMCI, AD]; label time=Year 2 [NC, MCI, AD]
    num_classes = 4 if opt.label_time == 'bl' else 3

    # Feature extraction from existing weights
    MRI, PET, Non_Img, Label, length = GetFeatures([train_loader, test_loader], model)
    # create hypergraph
    model.HGconstruct(MRI, PET, Non_Img)
    model.info(length)
    model.load_networks('latest')

    model.set_HGinput(Label)
    validation(model, 0, writer)
