import numpy as np
import torch
from torch.nn import functional as F
import pdb
import itertools
from tqdm import tqdm
from .base_model import BaseModel
from . import networks3D
from .densenet import *
from .hypergraph_utils import *
from .hypergraph import *
from utils import contrastive_loss

class HGIBSemiUnlabeledConsistencyModel(BaseModel):
    def name(self):
        return 'HGIBSemiUnlabeledConsistencyModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        class_num = 3
        self.K_neigs = opt.K_neigs
        self.beta = opt.beta
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.using_focalloss = opt.focal
        self.loss_names = ['cls']
        self.train_encoders = opt.train_encoders

        if self.using_focalloss:
            self.loss_names.append('focal')
        self.loss_names.append('kl')
        
        self.model_names = ['Encoder_MRI', 'Encoder_PET', 'Encoder_NonImage', 'Decoder_HGIB']

        self.netEncoder_MRI = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_PET = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_NonImage = networks3D.init_net_update(networks3D.Encoder_NonImage(in_channels=7, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        
        self.num_graph_update = opt.num_graph_update
        self.weight_u = opt.weight_u # loss weight for unlabeled data
        self.netClassifier = torch.nn.Linear(1024*3, class_num)
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netClassifier.to(self.gpu_ids[0])
            self.netClassifier = torch.nn.DataParallel(self.netClassifier, self.gpu_ids)
        self.netDecoder_HGIB = networks3D.init_net_update(HGIB_v1(1024*3, 1024, 3, use_bn=False, heads=1), self.gpu_ids)

        self.criterionCE = torch.nn.CrossEntropyLoss()
        self.contrastive_loss = contrastive_loss.ContrastiveLoss(opt.batch_size)
        
        # initialize optimizers
        if self.isTrain:
            self.optimizer = torch.optim.AdamW([{'params': self.netDecoder_HGIB.parameters()}, 
                                                {'params': self.netEncoder_MRI.parameters()}, 
                                                {'params': self.netEncoder_PET.parameters()}, 
                                                {'params': self.netEncoder_NonImage.parameters()},
                                                {'params': self.netClassifier.parameters()}
                                                ],
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers = []
            self.optimizers.append(self.optimizer)

    def info(self, lens):
        self.len_train = lens[0]
        self.len_test = lens[1]

    def set_input(self, input):
        self.MRI = input[0].to(self.device)
        self.PET = input[1].to(self.device)
        self.target = input[2].to(self.device)
        self.nonimage = input[3].to(self.device)

    def set_HGinput(self, input=None):
        self.embedding = self.embedding.to(self.device)
        if input is not None:
            self.target = input.to(self.device)

    def ExtractFeatures(self, phase='test'):
        if phase == 'test':
            with torch.no_grad():
                self.embedding_MRI = self.netEncoder_MRI(self.MRI)
                self.embedding_PET = self.netEncoder_PET(self.PET)
                self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        else:
            self.embedding_MRI = self.netEncoder_MRI(self.MRI)
            self.embedding_PET = self.netEncoder_PET(self.PET)
            self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        return self.embedding_MRI, self.embedding_PET, self.embedding_NonImage

    def HGconstruct(self, embedding_MRI, embedding_PET, embedding_NonImage):
        G = Hypergraph.from_feature_kNN(embedding_MRI, self.K_neigs, self.device)
        G.add_hyperedges_from_feature_kNN(embedding_PET, self.K_neigs)
        G.add_hyperedges_from_feature_kNN(embedding_NonImage, self.K_neigs)
        self.G = G  # construct graph for the forward pass
        self.embedding = torch.Tensor(np.hstack((embedding_MRI, embedding_PET, embedding_NonImage))).to(self.device)

    def forward(self, phase='train', train_loader=None, test_loader=None, train_loader_u=None, epoch=None):
        if phase == 'train':
            assert train_loader is not None, 'train_loader is None, please provide train_loader for training'
            len_train_loader = len(train_loader)
            train_loader_x_iter = iter(train_loader)
            if train_loader_u is not None:
                len_train_loader = max(len(train_loader), len(train_loader_u))
                train_loader_u_iter = iter(train_loader_u)
            for i in range(len_train_loader): 
                try:
                    data = next(train_loader_x_iter)
                except StopIteration:
                    train_loader_x_iter = iter(train_loader)
                    data = next(train_loader_x_iter)
                try:
                    data_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(train_loader_u)
                    data_u = next(train_loader_u_iter)
                if epoch is not None:
                    weight_u = self.weight_u * min(epoch / 80., 1.)
                else:
                    weight_u = self.weight_u
                self.set_input(data)
                self.ExtractFeatures(phase='train')
                embedding = torch.cat((self.embedding_MRI, self.embedding_PET, self.embedding_NonImage), dim=1)
                prediction = self.netClassifier(embedding)
                self.loss_cls = self.criterionCE(prediction, self.target)
                
                # create hypergraph
                self.HGconstruct(self.embedding_MRI.cpu().detach().numpy(), 
                                self.embedding_PET.cpu().detach().numpy(), 
                                self.embedding_NonImage.cpu().detach().numpy())
                self.info([self.embedding_MRI.size(0), 0])
                # the following statement is useless as self.target is not applicable to unlabeled data
                self.set_HGinput(self.target)
                #hyper_graph = self.G.drop_hyperedges(drop_rate=0.2)
                prediction_x_graph = self.netDecoder_HGIB(self.embedding,self.G)
                prediction_x_graph = F.softmax(prediction_x_graph[0][-1], 1)
                prediction_x = F.softmax(prediction, 1)
                loss_x = ((prediction_x_graph - prediction_x)**2).sum(1).mean()
                
                if self.using_focalloss:
                    gamma = 0.5
                    alpha = 2
                    pt = torch.exp(-self.loss_cls)
                    self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
                    self.loss = self.loss_cls + self.loss_focal
                else:
                    self.loss = self.loss_cls
                # Consistency regularization on unlabeled data
                MRI_np, PET_np, _, non_image, MRI_str_aug, PET_str_aug = data_u
                embedding_MRI = self.netEncoder_MRI(MRI_np)
                embedding_PET = self.netEncoder_PET(PET_np)
                
                embedding_MRI_str_aug = self.netEncoder_MRI(MRI_str_aug)
                embedding_PET_str_aug = self.netEncoder_PET(PET_str_aug)
                
                loss_u = (self.contrastive_loss(embedding_MRI, embedding_MRI_str_aug) + self.contrastive_loss(embedding_PET, embedding_PET_str_aug)) + \
                    0.1*(self.contrastive_loss(embedding_MRI, embedding_PET_str_aug) + self.contrastive_loss(embedding_PET, embedding_MRI_str_aug))
                self.loss = self.loss + weight_u * loss_x + weight_u * loss_u
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                if i % 100 == 0:
                    print('Iteration {}, loss for encoders {}, loss_x {}, loss_u {}'.format(
                        i, self.loss.item(), loss_x.item(), loss_u.item()))
            MRI, PET, Non_Img, Label, length = self.get_features([train_loader, test_loader])
            # create hypergraph
            self.HGconstruct(MRI, PET, Non_Img)
            self.info(length)
            self.set_HGinput(Label)
            num_graph_update = self.num_graph_update
            idx = torch.tensor(range(self.len_train)).to(self.device)
            prediction_encoder = self.netClassifier(self.embedding)
            for i in range(num_graph_update):
                # prediction is  [y1, y5], (kl1 + kl5)/2.0
                #hyper_graph = self.G.drop_hyperedges(drop_rate=0.2)
                self.prediction = self.netDecoder_HGIB(self.embedding, self.G)
                self.loss_cls = 0
            
                weight = [0.5, 0.5]
                # self.prediction[0] = [y1, y5]
                for t, pred in enumerate(self.prediction[0]):
                    self.loss_cls += weight[t] * self.criterionCE(pred[idx], self.target[idx])

                self.loss_kl = self.prediction[1]
                # self.loss_kd = 0

                if self.using_focalloss:
                    gamma = 0.5
                    alpha = 2
                    pt = torch.exp(-self.loss_cls)
                    self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
                    self.loss = self.loss_cls + self.loss_focal
                else:
                    self.loss = self.loss_cls
                
                self.loss = self.loss + self.loss_kl * self.beta
                self.prediction_cur = self.prediction[0][-1][idx]
                self.target_cur = self.target[idx]
                self.pred_encoder = prediction_encoder[idx]
                self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
                self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
                    
                if (i % 20 == 0) or (i == (num_graph_update - 1)):
                    print('Update the hyper-graph net for the {} times, total loss {}'.format(i, self.loss.item()))
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
        elif phase == 'test':
            with torch.no_grad():
                MRI, PET, Non_Img, Label, length = self.get_features([train_loader, test_loader], phase=phase)
                # create hypergraph
                self.HGconstruct(MRI, PET, Non_Img)
                self.info(length)
                self.set_HGinput(Label)
                num_graph_update = self.num_graph_update
                idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
                #idx = torch.tensor(range(self.len_train)).to(self.device)
                prediction_encoder = self.netClassifier(self.embedding)
                # prediction is  [y1, y5], (kl1 + kl5)/2.0
                #hyper_graph = self.G.drop_hyperedges(drop_rate=0.2)
                self.prediction = self.netDecoder_HGIB(self.embedding, self.G)
                self.loss_cls = 0
                self.loss_kl = 0
                self.loss_focal = 0
                self.loss = self.loss_cls
                
                self.prediction_cur = self.prediction[0][-1][idx]
                self.target_cur = self.target[idx]
                self.pred_encoder = prediction_encoder[idx]
                print('testing: shape of self.target_cur {}, shape of self.target {}'.format(self.target_cur.shape, self.target.shape))
                self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
                self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
        else:
            print('Wrong in loss calculation')
            exit(-1)


    def optimize_parameters(self, train_loader, test_loader, train_loader_u=None, epoch=None):
        # forward pass is here
        self.netClassifier.train()
        self.train()
        self.forward('train', train_loader, test_loader, train_loader_u, epoch)

    def validation(self):
        self.netClassifier.eval()
        self.eval()
        with torch.no_grad():
            self.forward('test', self.train_loader, self.test_loader)

    def get_pred_encoder(self):
        return self.pred_encoder
    
    def get_acc_encoder(self):
        return self.acc_encoder
    
    def get_features(self, loaders, phase='test'):
        # extract featrues from pre-trained model
        # stack them
        MRI = None
        PET = None
        Non_Img = None
        Label = None
        length = [0, 0]
        for idx, loader in enumerate(loaders):
            for i, data in enumerate(tqdm(loader)):
                self.set_input(data)
                i_MRI, i_PET, i_Non_Img = self.ExtractFeatures(phase)
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
