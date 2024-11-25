import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
import matplotlib.pyplot as plt
import numpy as np
import pdb
from utils.NiftiDataset_cls_densenet_non_imaging_cross_val import NifitDataSetCrossVal
from utils.NiftiDataset_cls_semisup_non_imaging import NifitSemiSupDataSet
from options.train_options import TrainOptions
import time
from models import create_model
from utils.visualizer import Visualizer
from tqdm import tqdm
from monai.data import DataLoader
from torchmetrics import ConfusionMatrix, Accuracy
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassSpecificity, MulticlassRecall, MulticlassAUROC

from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    CenterSpatialCrop,
    RandAdjustContrast,
    RandFlip,
    RandRotate
)
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.utils import GetFeatures, cal_metrics

import random


def validation(model, epoch, writer=None):
    visualizer = Visualizer(opt)
    total_steps = 0
    print("Validating model...")

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
    acc_class_reduction = metric(prediction, target)
    metric = MulticlassAccuracy(num_classes=num_classes, average=None, thresholds=None)
    acc = metric(prediction, target)
    # if (acc_class_reduction - acc.mean()) != 0:
    #     print('HIB acc for all classes {}, per-class mean acc {}'.format(acc_class_reduction, acc.mean()))

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
    print('[Validation] loss: ', np.sum(losses, 0) / len(losses), '\t\t Accuracy:', acc_class_reduction,
          'Per-class Acc', acc, 'Mean Acc', acc.mean().item(), 'AUC', AUROC, 'Mean AUC:', AUROC.mean().item(), 
          'PPV:', PPV, 'Mean PPV:', PPV.mean(), 'NPV:', NPV, 'Mean NPV:', NPV.mean())
    if True:# try:
        pred_encoder = model.get_pred_encoder()
        pred_encoder = pred_encoder.cpu().detach()
        metric_encoder = MulticlassAccuracy(num_classes=num_classes, average='micro', thresholds=None)
        acc_pred_class_reduction = metric_encoder(pred_encoder, target)
        metric_encoder = MulticlassAccuracy(num_classes=num_classes, average=None, thresholds=None)
        acc_pred = metric_encoder(pred_encoder, target)
        # if (acc_pred_class_reduction - acc_pred.mean()) != 0:
        #     print('Discriminative classifier acc for all classes {}, per-class mean acc {}'.format(
        #         acc_pred_class_reduction, acc_pred.mean()))
        metric_encoder = MulticlassF1Score(num_classes=num_classes, average=None)
        F1 = metric_encoder(pred_encoder, target)
        metric_encoder = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None)
        AUROC_encoder = metric_encoder(pred_encoder, target)

        metric_encoder = MulticlassSpecificity(num_classes=num_classes, average=None)
        Specificity = metric_encoder(pred_encoder, target)
        metric_encoder = MulticlassRecall(num_classes=num_classes, average=None)
        recall = metric_encoder(pred_encoder, target)

        confmat_encoder = ConfusionMatrix(num_classes=num_classes)
        CM = confmat_encoder(pred_encoder, target)
        #save_image(CM.clone().detach().float(), 'visualization/confusion_matrix.png')
        PPV_encoder, NPV_encoder = cal_metrics(CM)
        print('[Validation] for encoders: ', '\t\t Accuracy:', acc_pred_class_reduction, 
            'Per-class Acc', acc_pred, 'Mean Acc:', acc_pred.mean().item(),
            'AUC', AUROC_encoder, 'Mean AUC:', AUROC_encoder.mean().item(), 'PPV:', PPV_encoder, 
            'Mean PPV:', PPV_encoder.mean(), 'NPV:', NPV_encoder, 'Mean NPV:', NPV_encoder.mean())
        fig = plt.figure()
        plt.imshow(CM)
        plt.colorbar()
        plt.savefig('fig/confusion_matrix.png')
        
        pred_avg = (pred_encoder + prediction) / 2.
        CM = confmat_encoder(pred_avg, target)
        PPV_avg, NPV_avg = cal_metrics(CM)
        metric_encoder = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None)
        AUROC_avg = metric_encoder(pred_avg, target)
        print('[Validation] Averaged Prediction: ', '\t\t AUC', AUROC_avg, 
            'Mean AUC:', AUROC_avg.mean().item(), 'PPV:', PPV_avg, 
            'Mean PPV:', PPV_avg.mean(), 'NPV:', NPV_avg, 'Mean NPV:', NPV_avg.mean())
    # except:
    #     pass

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    seed=opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    use_writer = False
    if use_writer:
        writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard/logs'))
    else:
        writer = None
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    # load dataset
    trainTransforms = Compose([RandFlip(prob=0.5), ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    train_set = NifitDataSetCrossVal(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True, 
                                     phase='train', label_time=opt.label_time, control = opt.control, split=opt.split, label_ratio=opt.label_ratio, fold=opt.fold)
    print('length labeled train list:', len(train_set))
    if len(train_set) < opt.batch_size:
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers*len(opt.gpu_ids), 
                              pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g)  # Here are then fed to the network with a defined batch size
    # unlabeled data
    trainTransformsStrong = Compose([RandAdjustContrast(prob=0.3, gamma=(0.6, 1.5)), RandFlip(prob=0.5), 
                                     RandRotate(range_x=15.0, range_y=15.0, range_z=15.0, prob=0.5),
                                     ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    train_set_u = NifitSemiSupDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, 
                                      shuffle_labels=False, train=True, phase='train', label_time=opt.label_time, 
                                      control = opt.control, split=opt.split, use_strong_aug=opt.use_strong_aug,
                                      transforms_strong=trainTransformsStrong)
    train_loader_u = DataLoader(train_set_u, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                              num_workers=opt.workers*len(opt.gpu_ids), pin_memory=True, worker_init_fn=seed_worker, generator=g)
    testTransforms = Compose([ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    test_set = NifitDataSetCrossVal(opt.data_path, which_direction='AtoB', transforms=testTransforms, shuffle_labels=False,
                           test=True, phase='test', label_time=opt.label_time, control=opt.control, split=opt.split, fold=opt.fold)
    print('length test list:', len(test_set))
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True)

    # create model
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print("Creating model...")
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    if opt.load_weight:
        if not opt.continue_train:
            model.load_pretrained_networks(opt.epoch_count)
        else:
            print('opt.continue_train is True, expecting resuming training from epoch {}, \
                  but the weights at that epoch are over-written by {}'.format(opt.epoch_count, opt.load_weight))

    visualizer = Visualizer(opt)
    total_steps = 0
    # label time =bl [NC, EMCI, LMCI, AD]; label time=Year 2 [NC, MCI, AD]
    num_classes = 4 if opt.label_time == 'bl' else 3
    model.train_loader, model.test_loader = train_loader, test_loader
    MRI, PET, Non_Img, Label, length = GetFeatures([train_loader, test_loader], model)
    # create hypergraph
    model.HGconstruct(MRI, PET, Non_Img)
    model.info(length)
    # TODO: Add a classification head to encoders, maybe also enforce consistency between the classification head and the graph net's classification head

    print("Training model...")
    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1), desc="Current epoch during training."):
        if epoch > 1000:
            break
        epoch_iter = 0
        losses = []
        acc = []
        prediction = None
        target = None

        visualizer.reset()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        model.set_HGinput(Label)
        #if epoch == 0:
        #    validation(model, epoch*len(train_loader), writer)
        try:
            model.optimize_parameters(train_loader, test_loader, train_loader_u, epoch)
        except:
            if 'DHGNN' in opt.model:
                model.optimize_parameters(epoch)
            else:
                model.optimize_parameters()
        loss = model.get_current_losses()
        losses.append(loss)
        acc.append(model.get_current_acc())
        if use_writer:
            writer.add_scalar("train_loss/CE", loss[0],  epoch )
            if opt.focal:
                writer.add_scalar("train_loss/focal", loss[1], epoch)
            writer.add_scalar("train_loss/acc", model.get_current_acc(), epoch)
        prediction = model.get_prediction_cur()
        target = model.get_target_cur()
        prediction = prediction.cpu().detach()

        # evaluation results
        target = target.cpu().detach()
        metric = MulticlassAccuracy(num_classes=num_classes, average='micro', thresholds=None)
        acc = metric(prediction, target)
        metric = MulticlassF1Score(num_classes=num_classes, average=None)
        F1 = metric(prediction, target)
        metric = MulticlassAUROC(num_classes=num_classes, average=None, thresholds=None)
        AUROC = metric(prediction, target)
        print('Train loss: ', np.sum(losses, 0) / len(losses), '\t\t Accuracy:', acc, 'AUC', AUROC, 'F1', F1)

        if opt.focal:
            visualizer.print_val_losses(epoch, np.sum(losses, 0) / len(losses), acc, 'Train')
        else:
            visualizer.print_val_losses(epoch, losses, acc, 'Train')

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
            model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d ' % (epoch, opt.niter + opt.niter_decay))
        model.update_learning_rate(model.get_current_acc())

        if (epoch+1)%10 ==0:
            # TODO: remove the feature extraction part because extracting feature here can cause the performance drop
            MRI, PET, Non_Img, Label, length = GetFeatures([train_loader, test_loader], model)
            # create hypergraph
            model.HGconstruct(MRI, PET, Non_Img)
            model.info(length)
            model.set_HGinput(Label)
            validation(model, epoch*len(train_loader), writer)
        if use_writer:
            writer.close()
