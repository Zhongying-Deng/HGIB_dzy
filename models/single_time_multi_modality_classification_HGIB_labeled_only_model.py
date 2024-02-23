import numpy as np
import torch
import itertools
from .base_model import BaseModel
from . import networks3D
from .densenet import *
from .hypergraph_utils import *
from .hypergraph import *
import pdb

class SingleTimeMultiModalityClassificationHGIBLabeledOnlyModel(BaseModel):
    def name(self):
        return 'SingleTimeMultiModalityClassificationHGIBLabeledOnlyModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.K_neigs = opt.K_neigs
        self.beta = opt.beta
        # current input size is [121, 145, 121]
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.using_focalloss = opt.focal
        self.loss_names = ['cls']


        if self.using_focalloss:
            self.loss_names.append('focal')
        self.loss_names.append('kl')
        # self.loss_names.append('kd')

        self.model_names = ['Encoder_MRI', 'Encoder_PET', 'Encoder_NonImage', 'Decoder_HGIB']

        self.netEncoder_MRI = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_PET = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_NonImage = networks3D.init_net_update(networks3D.Encoder_NonImage(in_channels=7, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netDecoder_HGIB = networks3D.init_net_update(HGIB_v1(1024*3, 1024, 3, use_bn=False, heads=1), self.gpu_ids)
        self.netClassifier = torch.nn.Linear(1024*3, 3)
        self.use_modal_cls = False
        
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netClassifier.to(self.gpu_ids[0])
            self.netClassifier = torch.nn.DataParallel(self.netClassifier, self.gpu_ids)
            
        self.criterionCE = torch.nn.CrossEntropyLoss()
        self.num_graph_update = opt.num_graph_update
        self.weight_u = opt.weight_u
        # initialize optimizers
        if self.isTrain:
            # Why not update encoders?
            #self.optimizer = torch.optim.Adam(itertools.chain(self.netDecoder_HGIB.parameters()), #, self.netEncoder_MRI.parameters(), self.netEncoder_PET.parameters(), self.netEncoder_NonImage.parameters(), ),
            #                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer = torch.optim.Adam([{'params': self.netDecoder_HGIB.parameters()}, 
                                                {'params': self.netClassifier.parameters()},
                                                #{'params': self.netEncoder_MRI.parameters()}, 
                                                #{'params': self.netEncoder_PET.parameters()}, 
                                                #{'params': self.netEncoder_NonImage.parameters()},
                                                ],
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer)

    def info(self, lens):
        self.len_train = lens[0]
        self.len_test = lens[1]

    def set_input(self, input, update_target=True):
        self.MRI = input[0].to(self.device)
        self.PET = input[1].to(self.device)
        if update_target:
            self.target = input[2].to(self.device)
        self.nonimage = input[3].to(self.device)

    def set_HGinput(self, input=None):
        self.embedding = self.embedding.to(self.device)
        if input is not None:
            self.target = input.to(self.device)

    def ExtractFeatures(self, phase='test'):
        #if phase == 'test':
        if True:
            with torch.no_grad():
                self.embedding_MRI = self.netEncoder_MRI(self.MRI)
                self.embedding_PET = self.netEncoder_PET(self.PET)
                self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        #else:
        #    self.embedding_MRI = self.netEncoder_MRI(self.MRI)
        #    self.embedding_PET = self.netEncoder_PET(self.PET)
        #    self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        return self.embedding_MRI, self.embedding_PET, self.embedding_NonImage

    def HGconstruct(self, embedding_MRI, embedding_PET, embedding_NonImage):
        G = Hypergraph.from_feature_kNN(embedding_MRI, self.K_neigs, self.device)
        G.add_hyperedges_from_feature_kNN(embedding_PET, self.K_neigs)
        G.add_hyperedges_from_feature_kNN(embedding_NonImage, self.K_neigs)
        self.G = G  # construct graph for the forward pass
        self.embedding = torch.Tensor(np.hstack((embedding_MRI, embedding_PET, embedding_NonImage))).to(self.device)

    def forward(self, phase='train', train_loader=None, test_loader=None, train_loader_u=None, epoch=None):
        if phase == 'train':
            if False:
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
                    self.set_input(data)
                    self.ExtractFeatures(phase='train')
                    embedding = torch.cat((self.embedding_MRI, self.embedding_PET, self.embedding_NonImage), dim=1)
                    # embedding = self.embedding_MRI + self.embedding_PET + self.embedding_NonImage
                    prediction = self.netClassifier(embedding)
                    self.loss_cls = self.criterionCE(prediction, self.target)
                    if self.using_focalloss:
                        gamma = 0.5
                        alpha = 2
                        pt = torch.exp(-self.loss_cls)
                        self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
                        self.loss = self.loss_focal
                    else:
                        self.loss = self.loss_cls
                    self.set_input(data_u, update_target=False)
                    self.ExtractFeatures(phase='train')
                    embedding_u = torch.cat((self.embedding_MRI, self.embedding_PET, self.embedding_NonImage), dim=1)
                    # embedding = self.embedding_MRI + self.embedding_PET + self.embedding_NonImage
                    prediction_u = self.netClassifier(embedding_u)
                    # create hypergraph
                    self.HGconstruct(self.embedding_MRI.cpu().detach().numpy(), 
                                    self.embedding_PET.cpu().detach().numpy(), 
                                    self.embedding_NonImage.cpu().detach().numpy())
                    self.info([self.embedding_MRI.size(0), 0])
                    # the following statement is useless as self.target is not applicable to unlabeled data
                    self.set_HGinput(self.target)
                    prediction_u_graph = self.netDecoder_HGIB(self.embedding, self.G)
                    prediction_u_graph = F.softmax(prediction_u_graph[0][-1], 1)
                    prediction_u = F.softmax(prediction_u, 1)
                    loss_u = ((prediction_u_graph - prediction_u)**2).sum(1).mean()
                    if epoch is not None:
                        weight_u = self.weight_u * min(epoch / 80., 1.)
                    else:
                        weight_u = self.weight_u
                    self.loss = self.loss + weight_u * loss_u
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()
                    if i % 100 == 0:
                        print('Iteration {}, loss for encoders {}, loss_u {}'.format(i, self.loss.item(), loss_u.item()))
                MRI, PET, Non_Img, Label, length = self.get_features([train_loader, test_loader])
                # create hypergraph
                self.HGconstruct(MRI, PET, Non_Img)
                self.info(length)
                self.set_HGinput(Label)
            num_graph_update = self.num_graph_update
            idx = torch.tensor(range(self.len_train)).to(self.device)
        elif phase == 'test':
            num_graph_update = 1
            idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
        else:
            print('Wrong in loss calculation')
            exit(-1)

        
        for i in range(num_graph_update):
            prediction_encoder = self.netClassifier(self.embedding)
            self.prediction = self.netDecoder_HGIB(self.embedding, self.G)
            self.loss_cls = self.criterionCE(prediction_encoder[idx], self.target[idx])
            #self.loss_cls = 0
            weight = [0.5, 0.5]
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
                self.loss = self.loss_cls #+ self.loss_cls
            if False:
                prediction_u_graph = F.softmax(self.prediction[0][-1], 1)
                prediction_u = F.softmax(prediction_encoder, 1)
                loss_u = ((prediction_u_graph - prediction_u)**2).sum(1).mean()
                if epoch is not None:
                    weight_u = self.weight_u * min(epoch / 80., 1.)
                else:
                    weight_u = self.weight_u
                self.loss = self.loss_cls + weight_u * loss_u + self.loss_kl * self.beta
            else:
                self.loss = self.loss_cls + self.loss_kl * self.beta
            if phase == 'train':
                if (i % 20 == 0) or (i == (num_graph_update - 1)):
                    print('Update the hyper-graph net for the {} times, total loss {}'.format(i, self.loss.item()))
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            self.prediction_cur = self.prediction[0][-1][idx]
            self.target_cur = self.target[idx]
            self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.pred_encoder = prediction_encoder[idx]
            self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))

    def optimize_parameters(self, train_loader=None, test_loader=None, train_loader_u=None, epoch=None):
        self.optimizer.zero_grad()
        # forward pass is here
        self.forward('train', train_loader, test_loader, train_loader_u=train_loader_u, epoch=epoch)
        #self.loss.backward()
        #self.optimizer.step()

    def validation(self):
        with torch.no_grad():
            self.forward('test')

    def get_pred_encoder(self):
        return self.pred_encoder

    def get_features(self, loaders):
        # extract featrues from pre-trained model
        # stack them 
        MRI = None
        PET = None
        Non_Img = None
        Label = None
        length = [0, 0]
        for idx, loader in enumerate(loaders):
            for i, data in enumerate(loader):
                self.set_input(data)
                i_MRI, i_PET, i_Non_Img = self.ExtractFeatures()
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
                if i % 10 == 0:
                    print('extract features for loader {}, iteration {}'.format(idx, i))
            length[idx] = MRI.size(0)
        length[1] = length[1] - length[0]
        return MRI.cpu().detach().numpy(), PET.cpu().detach().numpy(), Non_Img.cpu().detach().numpy(), Label, length
