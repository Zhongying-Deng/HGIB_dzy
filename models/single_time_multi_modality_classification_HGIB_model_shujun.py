import numpy as np
import torch
import itertools
from .base_model import BaseModel
from . import networks3D
from .densenet import *
from .hypergraph_utils import *
from .hypergraph import *

class SingleTimeMultiModalityClassificationHGIBModel(BaseModel):
    def name(self):
        return 'SingleTimeMultiModalityClassificationHGIBModel'

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

        self.criterionCE = torch.nn.CrossEntropyLoss()

        # initialize optimizers
        if self.isTrain:
            self.optimizer = torch.optim.Adam(itertools.chain(self.netDecoder_HGIB.parameters()), #, self.netEncoder_MRI.parameters(), self.netEncoder_PET.parameters(), self.netEncoder_NonImage.parameters(), ),
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

    def set_HGinput(self, input):
        self.embedding = self.embedding.to(self.device)
        self.target = input.to(self.device)

    def ExtractFeatures(self):
        with torch.no_grad():
            self.embedding_MRI = self.netEncoder_MRI(self.MRI)
            self.embedding_PET = self.netEncoder_PET(self.PET)
            self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        return self.embedding_MRI, self.embedding_PET, self.embedding_NonImage

    def HGconstruct(self, embedding_MRI, embedding_PET, embedding_NonImage):
        G = Hypergraph.from_feature_kNN(embedding_MRI, self.K_neigs, self.device)
        G.add_hyperedges_from_feature_kNN(embedding_PET, self.K_neigs)
        G.add_hyperedges_from_feature_kNN(embedding_NonImage, self.K_neigs)
        self.G = G
        self.embedding = torch.Tensor(np.hstack((embedding_MRI, embedding_PET, embedding_NonImage))).to(self.device)

    def forward(self, phase='train'):
        self.prediction = self.netDecoder_HGIB(self.embedding, self.G)
        if phase == 'train':
            idx = torch.tensor(range(self.len_train)).to(self.device)
        elif phase == 'test':
            idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
        else:
            print('Wrong in loss calculation')
            exit(-1)

        self.loss_cls = 0
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
            self.loss = self.loss_cls
        self.loss= self.loss_cls + self.loss_kl * self.beta

        self.prediction_cur = self.prediction[0][-1][idx]
        self.target_cur = self.target[idx]
        self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward('train')
        self.loss.backward()
        self.optimizer.step()

    def validation(self):
        with torch.no_grad():
            self.forward('test')


