# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import copy
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from .layer_utils import *

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    print(type(H))
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x


class HGNNPConv(nn.Module):
    r"""The HGNN :sup:`+` convolution layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Sparse Format:

    .. math::

        \left\{
            \begin{aligned}
                m_{\beta}^{t} &=\sum_{\alpha \in \mathcal{N}_{v}(\beta)} M_{v}^{t}\left(x_{\alpha}^{t}\right) \\
                y_{\beta}^{t} &=U_{e}^{t}\left(w_{\beta}, m_{\beta}^{t}\right) \\
                m_{\alpha}^{t+1} &=\sum_{\beta \in \mathcal{N}_{e}(\alpha)} M_{e}^{t}\left(x_{\alpha}^{t}, y_{\beta}^{t}\right) \\
                x_{\alpha}^{t+1} &=U_{v}^{t}\left(x_{\alpha}^{t}, m_{\alpha}^{t+1}\right) \\
            \end{aligned}
        \right.

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e
        \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.v2v(X, aggr="mean")
        if not self.is_last:
            X = self.drop(self.act(X))
        return X


class HGNNPConv_GIB_v1(nn.Module):
    r"""The HGNN :sup:`+` convolution layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
    Sparse Format:
    .. math::

        \left\{
            \begin{aligned}
                m_{\beta}^{t} &=\sum_{\alpha \in \mathcal{N}_{v}(\beta)} M_{v}^{t}\left(x_{\alpha}^{t}\right) \\
                y_{\beta}^{t} &=U_{e}^{t}\left(w_{\beta}, m_{\beta}^{t}\right) \\
                m_{\alpha}^{t+1} &=\sum_{\beta \in \mathcal{N}_{e}(\alpha)} M_{e}^{t}\left(x_{\alpha}^{t}, y_{\beta}^{t}\right) \\
                x_{\alpha}^{t+1} &=U_{v}^{t}\left(x_{\alpha}^{t}, m_{\alpha}^{t+1}\right) \\
            \end{aligned}
        \right.

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e
        \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
            heads: int = 8,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.att = Parameter(torch.Tensor(heads, out_channels))
        self.heads = heads
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(1))
        self.att.data.uniform_(-stdv, stdv)

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        structure_kl_loss = 0.0
        X = self.theta(X)   # torch.Size([203, 1024])
        # print("x size:", X.size())
        if self.bn is not None:
            X = self.bn(X)
        # message passing to other nodes
        X = hg.v2v(X, aggr="mean")    # torch.Size([203, 1024])
        # print("X_1 size:", X_1.size())
        if not self.is_last:
            X = self.drop(self.act(X))

            # multi-head attention
            # Ze  = hg.v2e(X, aggr="mean")    # Eq.14 mean update the edge features based on the updated vertex features  torch.Size([599, 1024])
            # print('Ze.size():', Ze.size())
            X_1 = X.view(-1, self.heads, self.out_channels)
            # attention scores according to GIB paper
            alpha = (X_1 * self.att).mean(dim=-1).view(-1) # torch.Size([29, 7])
            # print("alpha size:", alpha.size())
            alpha = F.leaky_relu(alpha, 0.2)
            # alpha_normalization = torch.ones_like(alpha)
            # alpha_normalization = F.softmax(alpha, -1)
            # alpha = alpha * alpha_normalization
            alpha = torch.clamp(torch.sigmoid(alpha), 0.01, 0.99)
            self.alpha = alpha
            self.prior = (torch.ones_like(self.alpha) * 0.5).to(alpha.device)   # 0.5

            posterior = torch.distributions.bernoulli.Bernoulli(self.alpha)
            prior = torch.distributions.bernoulli.Bernoulli(self.prior)
            structure_kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
        return X, structure_kl_loss

class HGNNPConv_GIB_v2(nn.Module):
    r"""The HGNN :sup:`+` convolution layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
    Sparse Format:
    .. math::

        \left\{
            \begin{aligned}
                m_{\beta}^{t} &=\sum_{\alpha \in \mathcal{N}_{v}(\beta)} M_{v}^{t}\left(x_{\alpha}^{t}\right) \\
                y_{\beta}^{t} &=U_{e}^{t}\left(w_{\beta}, m_{\beta}^{t}\right) \\
                m_{\alpha}^{t+1} &=\sum_{\beta \in \mathcal{N}_{e}(\alpha)} M_{e}^{t}\left(x_{\alpha}^{t}, y_{\beta}^{t}\right) \\
                x_{\alpha}^{t+1} &=U_{v}^{t}\left(x_{\alpha}^{t}, m_{\alpha}^{t+1}\right) \\
            \end{aligned}
        \right.

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e
        \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
            heads: int = 8,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.att = Parameter(torch.Tensor(heads, out_channels))
        self.heads = heads
        self.out_channels = out_channels
        self.reset_parameters()
        self.alpha = 0.5
        self.threshold = 0.4

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(1))
        self.att.data.uniform_(-stdv, stdv)


    def forward(self, X, hg):
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        structure_kl_loss = 0.0
        X = self.theta(X)   # torch.Size([203, 1024])
        # print("x size:", X.size())
        if self.bn is not None:
            X = self.bn(X)
        # message passing to other nodes
        X = hg.v2v(X, aggr="mean")    # torch.Size([203, 1024])
        # print("X_1 size:", X_1.size())
        if not self.is_last:
            X = self.drop(self.act(X))

            # multi-head attention
            Ze  = hg.v2e(X, aggr="sum")    # Eq.14 mean update the edge features based on the updated vertex features  torch.Size([599, 1024]) 599 is the edge number in hypergraph
            # print('Ze.size():', Ze.size(), (X_1 * self.att).size(),  self.heads, self.out_channels)
            # Ze = Ze.view(-1, self.heads, self.out_channels)
            X_1 = X.view(-1, self.heads, self.out_channels) # [203, 1, 1024] [31,8,1024]
            # print('Ze.size():', Ze.size(), (X_1 * self.att).size(), self.att.size())
            # attention in hypergraph structure paper
            COS_SIM = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            # A = COS_SIM(X_1, Ze)
            A = COS_SIM((X_1 * self.att), Ze * self.att) # A size: torch.Size([203, 1, 599])
            A = A.view(-1, A.size()[-1])    # A size: torch.Size([203, 599])
            H = hg.H.to_dense()     # hypergraph edge matrics H size: torch.Size([203, 599])
            updated_H = self.alpha * H + (1-self.alpha) * A
            # print('A:', A)
            # updated_H = self.alpha * H + 1 * A
            # updated_H = torch.sigmoid(updated_H)
            # updated_H = updated_H.float()*1.0
            updated_H = (updated_H > self.threshold).float()
            hg.cache["H"] = updated_H.to_sparse()

            self.prior = (torch.ones_like(A) * 0.5).to(A.device)  # 0.5
            posterior = torch.distributions.bernoulli.Bernoulli(A)
            prior = torch.distributions.bernoulli.Bernoulli(self.prior)
            structure_kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
            # alpha = alpha.view(-1)
            # X =
            # X = X_1
        return X, structure_kl_loss

class HGNNPConv_GIB_v3(nn.Module):
    r"""The HGNN :sup:`+` convolution layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
    Sparse Format:
    .. math::

        \left\{
            \begin{aligned}
                m_{\beta}^{t} &=\sum_{\alpha \in \mathcal{N}_{v}(\beta)} M_{v}^{t}\left(x_{\alpha}^{t}\right) \\
                y_{\beta}^{t} &=U_{e}^{t}\left(w_{\beta}, m_{\beta}^{t}\right) \\
                m_{\alpha}^{t+1} &=\sum_{\beta \in \mathcal{N}_{e}(\alpha)} M_{e}^{t}\left(x_{\alpha}^{t}, y_{\beta}^{t}\right) \\
                x_{\alpha}^{t+1} &=U_{v}^{t}\left(x_{\alpha}^{t}, m_{\alpha}^{t+1}\right) \\
            \end{aligned}
        \right.

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e
        \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
            heads: int = 8,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.att = Parameter(torch.Tensor(heads, out_channels))
        self.heads = heads
        self.out_channels = out_channels

        self.alpha = 0.5
        self.threshold = 0.9
        # self.threshold = Parameter(torch.Tensor(203, 599))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.att.size(1))
        self.att.data.uniform_(-stdv, stdv)
        # self.threshold.data.uniform_(0, 1)

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        structure_kl_loss = 0.0
        X = self.theta(X)  # torch.Size([203, 1024])
        # print("x size:", X.size())
        if self.bn is not None:
            X = self.bn(X)
        # message passing to other nodes
        X = hg.v2v(X, aggr="mean")  # torch.Size([203, 1024])
        # print("X_1 size:", X_1.size())
        if not self.is_last:
            X = self.drop(self.act(X))

            # multi-head attention
            Ze = hg.v2e(X, aggr="mean")  # Eq.14 mean update the edge features based on the updated vertex features  torch.Size([599, 1024]) 599 is the edge number in hypergraph
            # print('Ze.size():', Ze.size())
            X_1 = X.view(-1, self.heads, self.out_channels)  # [203, 1, 1024]
            # attention in hypergraph structure paper
            COS_SIM = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            # A = COS_SIM(X_1, Ze)
            A = COS_SIM((X_1 * self.att), Ze * self.att)  # A size: torch.Size([203, 1, 599])
            A = A.view(-1, A.size()[-1])  # A size: torch.Size([203, 599])
            # print('A:', A.sum(), A.max(), A.min())
            # A = (A > self.threshold).float()

            H = hg.H.to_dense()  # hypergraph edge matrics H size: torch.Size([203, 599])
            # print('H:', H.sum())
            updated_H = self.alpha * H + (1 - self.alpha) * A

            # updated_H = self.alpha * H + 1 * A
            # updated_H = torch.sigmoid(updated_H)
            # updated_H = updated_H.float()*1.0

            # print(self.threshold)

            self.prior = (torch.ones_like(updated_H) * 0.5).to(updated_H.device)  # 0.5
            posterior = torch.distributions.bernoulli.Bernoulli(updated_H)
            prior = torch.distributions.bernoulli.Bernoulli(self.prior)
            structure_kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()

            updated_H = (updated_H >= 0.65).float()
            hg.cache["H"] = updated_H.to_sparse()
            # alpha = alpha.view(-1)
            # X =
            # X = X_1
        return X, structure_kl_loss, hg

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid=1024, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        # self.hgc2 = HGNN_conv(n_hid, n_hid)
        # self.hgc3 = HGNN_conv(n_hid, n_hid)
        # self.hgc4 = HGNN_conv(n_hid, n_hid)
        self.hgc5 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        # x = F.relu(self.hgc2(x, G))
        # x = F.dropout(x, self.dropout)
        #
        # x = F.relu(self.hgc3(x, G))
        # x = F.dropout(x, self.dropout)
        # x = F.relu(self.hgc4(x, G))
        # x = F.dropout(x, self.dropout)
        x = self.hgc5(x, G)


        return x

class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        N, k, _ = region_feats.size()  # (N, k, d) (203, 20, 256)
        # print(N, k, region_feats.size())
        conved = self.convKK(region_feats)  # (N, k*k, 1) 203, 400, 247
        # print(conved.size())
        multiplier = conved.view(N, k, k)  # (N, k, k)
        multiplier = self.activation(multiplier)  # softmax along last dimension
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats

class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    """

    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension
        :param k: k neighbors
        """
        super().__init__()

        self.trans = Transform(dim_in, k)  # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)  # (N, k, d) -> (N, 1, d)

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        # print('region_feats',region_feats)
        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)  # (N, 1, d)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats

class GraphConvolution(nn.Module):
    """
    A GCN layer
    """

    def __init__(self, dim_in=256,
            dim_out=3,
            dropout_rate=0.5,
            activation=None,
            structured_neighbor=20,
            nearest_neighbor=20,
            cluster_neighbor=20,
            wu_knn=0,
            wu_kmeans=10,
            wu_struct=5,
            n_cluster=50, # may change
            n_center=1,
            has_bias=True):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation
        """
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=has_bias)
        self.dropout = nn.Dropout(p=0.5)
        self.activation = activation

    def _region_aggregate(self, feats, edge_dict):
        N = feats.size()[0]
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])

        return pooled_feats

    def forward(self, ids, feats, edge_dict, G, ite):
        """
        :param ids: compatible with `MultiClusterConvolution`
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats  # (N, d)
        x = self.dropout(self.activation(self.fc(x)))  # (N, d')
        x = self._region_aggregate(x, edge_dict)  # (N, d)
        return x

class EdgeConv(nn.Module):
    """
    A Hyperedge Convolution layer
    Using self-attention to aggregate hyperedges
    """

    def __init__(self, dim_ft, hidden):
        """
        :param dim_ft: feature dimension
        :param hidden: number of hidden layer neurons
        """
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim_ft, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, ft):
        """
        use self attention coefficient to compute weighted average on dim=-2
        :param ft (N, t, d)
        :return: y (N, d)
        """
        scores = []
        n_edges = ft.size(1)
        for i in range(n_edges):
            scores.append(self.fc(ft[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)
        # print(scores.size(), ft.size())
        return (scores * ft).sum(1)

class DHGLayer(GraphConvolution):
    """
    A Dynamic Hypergraph Convolution Layer
    """

    def __init__(self, dim_in=256,
            dim_out=3,
            dropout_rate=0.5,
            activation=None,
            structured_neighbor=16,
            nearest_neighbor=16,
            cluster_neighbor=16,
            wu_knn=0,
            wu_kmeans=10,
            wu_struct=5,
            n_cluster=50, # may change
            n_center=1,
            has_bias=True):
        super().__init__()

        self.ks = structured_neighbor  # number of sampled nodes in graph adjacency
        self.n_cluster = n_cluster  # number of clusters
        self.n_center = n_center  # a node has #n_center adjacent clusters
        self.kn = nearest_neighbor  # number of the 'k' in k-NN
        self.kc = cluster_neighbor  # number of sampled nodes in a adjacent k-means cluster
        self.wu_knn = wu_knn
        self.wu_kmeans = wu_kmeans
        self.wu_struct = wu_struct

        self.vc_sn = VertexConv(self.dim_in, self.ks + self.kn)  # structured trans
        self.vc_s = VertexConv(self.dim_in, self.ks)  # structured trans
        self.vc_n = VertexConv(self.dim_in, self.kn)  # nearest trans
        self.vc_c = VertexConv(self.dim_in, self.kc)  # k-means cluster trans
        self.ec = EdgeConv(self.dim_in, hidden=self.dim_in // 4)
        self.kmeans = None
        self.structure = None
        self.activation=activation

    def _vertex_conv(self, func, x):
        return func(x)

    def _structure_select(self, ids, feats, edge_dict):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :param edge_dict: torch.LongTensor
        :return: mapped graph neighbors
        """
        if self.structure is None:
            _N = feats.size(0)
            idx = torch.LongTensor([sample_ids(edge_dict[i], self.ks) for i in range(_N)])  # (_N, ks)
            self.structure = idx
        else:
            idx = self.structure

        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        region_feats = feats[idx.view(-1)].view(N, self.ks, d)  # (N, ks, d)
        return region_feats

    def _nearest_select(self, ids, feats):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: mapped nearest neighbors
        """
        dis = cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=1)
        idx = idx[ids]
        N = len(idx)
        d = feats.size(1)
        nearest_feature = feats[idx.view(-1)].view(N, self.kn, d)  # (N, kn, d)
        return nearest_feature

    def _cluster_select(self, ids, feats):
        """
        compute k-means centers and cluster labels of each node
        return top #n_cluster nearest cluster transformed features
        :param ids: indices selected during train/valid/test, torch.LongTensor
        :param feats:
        :return: top #n_cluster nearest cluster mapped features
        """
        if self.kmeans is None:
            _N = feats.size(0)
            np_feats = feats.detach().cpu().numpy()
            # kmeans = KMeans(n_clusters=self.n_cluster, random_state=0, n_jobs=-1).fit(np_feats)
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np_feats)

            centers = kmeans.cluster_centers_
            dis = euclidean_distances(np_feats, centers)
            _, cluster_center_dict = torch.topk(torch.Tensor(dis), self.n_center, largest=False)
            cluster_center_dict = cluster_center_dict.numpy()
            point_labels = kmeans.labels_
            point_in_which_cluster = [np.where(point_labels == i)[0] for i in range(self.n_cluster)]
            idx = torch.LongTensor([[sample_ids_v2(point_in_which_cluster[cluster_center_dict[point][i]], self.kc)
                                     for i in range(self.n_center)] for point in range(_N)])  # (_N, n_center, kc)
            self.kmeans = idx
        else:
            idx = self.kmeans

        idx = idx[ids]
        N = idx.size(0)
        d = feats.size(1)
        cluster_feats = feats[idx.view(-1)].view(N, self.n_center, self.kc, d)

        return cluster_feats  # (N, n_center, kc, d)

    def _edge_conv(self, x):
        return self.ec(x)

    def _fc(self, x):
        return self.activation(self.fc(self.dropout(x)))

    def forward(self, ids, feats, edge_dict, G, ite):
        hyperedges = []
        if ite >= self.wu_kmeans:
            c_feat = self._cluster_select(ids, feats)
            for c_idx in range(c_feat.size(1)):
                xc = self._vertex_conv(self.vc_c, c_feat[:, c_idx, :, :])
                xc = xc.view(len(ids), 1, feats.size(1))  # (N, 1, d)
                hyperedges.append(xc)
        if ite >= self.wu_knn:
            n_feat = self._nearest_select(ids, feats)
            xn = self._vertex_conv(self.vc_n, n_feat)
            xn = xn.view(len(ids), 1, feats.size(1))  # (N, 1, d)
            hyperedges.append(xn)
        if ite >= self.wu_struct:
            s_feat = self._structure_select(ids, feats, edge_dict)
            xs = self._vertex_conv(self.vc_s, s_feat)
            xs = xs.view(len(ids), 1, feats.size(1))  # (N, 1, d)
            hyperedges.append(xs)
        x = torch.cat(hyperedges, dim=1)
        # print(x.size())
        x = self._edge_conv(x)  # (N, d)
        # print(x.size())
        x = self._fc(x)  # (N, d')
        return x

class DHGNN_conv(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, dim_in, dim_out, dropout_rate, activation, has_bias):
        super(DHGNN_conv, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=has_bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = activation


    def forward(self, ids, feats, edge_dict, G, ite):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x

class HGNNP(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class HGNNP_robust(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        hg = hg.drop_hyperedges(0.2)
        for layer in self.layers:
            X = layer(X, hg)
        return X


class HGNNPv2(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, hid_channels//2, use_bn=use_bn, is_last=True)
        )
        self.layers.append(
            HGNNPConv(hid_channels//2, hid_channels // 4, use_bn=use_bn, is_last=True)
        )
        self.layers.append(
            HGNNPConv(hid_channels//4, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X

class DHGNN(nn.Module):
    """
    Dynamic Hypergraph Convolution Neural Network with a HGNN-style input layer
    """
    def __init__(self, ):
        super().__init__()

        self.dim_feat = 1024*3
        self.n_categories = 3
        self.n_layers = 2
        layer_spec = [256]
        self.dims_in = [self.dim_feat] + layer_spec
        self.dims_out = layer_spec + [self.n_categories]
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])
        self.gcs = nn.ModuleList([DHGNN_conv(
            dim_in=self.dim_feat,
            dim_out=self.dims_out[0],
            dropout_rate=0.5,
            activation=activations[0],
            has_bias=True)]
            + [DHGLayer(
            dim_in=self.dims_in[i],
            dim_out=self.dims_out[i],
            dropout_rate=0.5,
            activation=activations[i],
            structured_neighbor=32,
            nearest_neighbor=32,
            cluster_neighbor=32,
            wu_knn=0,
            wu_kmeans=200,
            wu_struct=200,
            n_cluster=50, # may change
            n_center=1,
            has_bias=True) for i in range(1, self.n_layers)])

    def forward(self, feats, G, edge_dict, phase='train',epoch=1):
        """
        :param feats:
        :param edge_dict:
        :param G:
        :return:
        """
        ids = torch.tensor(range(248)).cuda()
        # if phase=='train':
        #     ids = torch.tensor(range(200)).cuda()
        # elif phase=='test':
        #     ids = torch.tensor(range(48)).cuda() + 200

        feats = feats
        edge_dict = edge_dict
        G = G
        ite = epoch

        x = feats
        for i_layer in range(self.n_layers):
            x = self.gcs[i_layer](ids, x, edge_dict, G, ite)
        return x

class HGIB_v2(nn.Module):
    '''
    This function is for hypergraph information bottelneck.
    using two cascated convolution layer
    there exists kl loss calucation within the convolution layer
    '''
    def __init__(self, in_ch, n_hid=1024, n_class=3, dropout=0.5, L=2, alpha=0.5, beta=0.01, theta=0.1, use_bn = False,
        drop_rate = 0.5, heads=8):
        super(HGIB_v2, self).__init__()
        self.dropout = dropout
        self.hgc = nn.ModuleList()

        self.conv1 = HGNNPConv_GIB_v2(in_ch, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv2 = HGNNPConv_GIB_v2(n_hid, n_class, use_bn=use_bn, is_last=True)

    def forward(self, x, G):
        x, kl = self.conv1(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        y, kl2 = self.conv2(x, G)
        return y, (kl +kl2)/2.0

class HGIB_v4(nn.Module):
    '''
    This function is for hypergraph information bottelneck.
    using two cascated convolution layer
    there exists kl loss calucation within the convolution layer
    '''
    def __init__(self, in_ch, n_hid=1024, n_class=3, dropout=0.5, L=2, alpha=0.5, beta=0.01, theta=0.1, use_bn = False,
        drop_rate = 0.5, heads=8):
        super(HGIB_v4, self).__init__()
        self.dropout = dropout
        self.hgc = nn.ModuleList()

        self.conv1 = HGNNPConv_GIB_v2(in_ch, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv1_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)
        self.conv2 = HGNNPConv_GIB_v2(n_hid, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)

        self.conv2_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)

    def forward(self, x, G):
        x, kl1 = self.conv1(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        y1 = self.conv1_1(x, G)

        x = F.relu(x)
        x = F.dropout(x, self.dropout)

        x, kl2 = self.conv2(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        y2 = self.conv2_1(x, G)


        return [y1, y2], (kl1 +kl2)/2.0

class HGIB_v3(nn.Module):
    '''
    This function is for hypergraph information bottelneck.
    using two cascated convolution layer
    there exists kl loss calucation within the convolution layer
    '''
    def __init__(self, in_ch, n_hid=1024, n_class=3, dropout=0.5, L=2, alpha=0.5, beta=0.01, theta=0.1, use_bn = False,
        drop_rate = 0.5, heads=8):
        super(HGIB_v3, self).__init__()
        self.dropout = dropout
        self.hgc = nn.ModuleList()

        self.conv1 = HGNNPConv_GIB_v3(in_ch, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv1_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)
        self.conv2 = HGNNPConv_GIB_v3(n_hid, n_hid//2, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv2_1 = HGNNPConv(n_hid//2, n_class, use_bn=use_bn, is_last=True)

    def forward(self, x, G):
        x, kl, G = self.conv1(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)

        y1 = self.conv1_1(x, G)
        x, kl2, G = self.conv2(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)

        y2 = self.conv2_1(x, G)
        return [y1, y2], (kl+kl2)/2


class HGIB_v1(nn.Module):
    '''
    This function is for hypergraph information bottelneck.
    using two cascated convolution layer
    there exists kl loss calucation within the convolution layer
    '''
    def __init__(self, in_ch, n_hid=1024, n_class=3, dropout=0.5, L=2, alpha=0.5, beta=0.01, theta=0.1, use_bn = False,
        drop_rate = 0.5, heads=8):
        super(HGIB_v1, self).__init__()
        self.dropout = dropout
        self.hgc = nn.ModuleList()

        self.conv1 = HGNNPConv_GIB_v1(in_ch, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv1_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)

        # self.conv2 = HGNNPConv_GIB_v1(n_hid, n_hid*2, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv2_1 = HGNNPConv(n_hid*2, n_class, use_bn=use_bn, is_last=True)

        # self.conv3 = HGNNPConv_GIB_v1(n_hid*2, n_hid*4, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv3_1 = HGNNPConv(n_hid*4, n_class, use_bn=use_bn, is_last=True)

        # self.conv4 = HGNNPConv_GIB_v1(n_hid * 4, n_hid * 4, use_bn=use_bn, drop_rate=drop_rate, is_last=False,
        #                               heads=heads)
        # self.conv4_1 = HGNNPConv(n_hid * 4, n_class, use_bn=use_bn, is_last=True)
        # self.conv5 = HGNNPConv_GIB_v1(n_hid * 4, n_hid * 4, use_bn=use_bn, drop_rate=drop_rate, is_last=False,
        #                               heads=heads)
        # self.conv5_1 = HGNNPConv(n_hid * 4, n_class, use_bn=use_bn, is_last=True)

        # self.conv4 = HGNNPConv_GIB_v1(n_hid * 4, n_hid*2, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv4_1 = HGNNPConv(n_hid*2, n_class, use_bn=use_bn, is_last=True)

        self.conv5 = HGNNPConv_GIB_v1(n_hid, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        self.conv5_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)


    def forward(self, x, G):
        x, kl1 = self.conv1(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        y1 = self.conv1_1(x, G)
        # x, kl2 = self.conv2(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y2 = self.conv2_1(x, G)
        # x, kl3 = self.conv3(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y3 = self.conv3_1(x, G)
        # x, kl4 = self.conv4(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y4 = self.conv4_1(x, G)
        x, kl5 = self.conv5(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        y5 = self.conv5_1(x, G)
        # x, kl6 = self.conv6(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y6 = self.conv6_1(x, G)
        # x, kl7 = self.conv7(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y7 = self.conv7_1(x, G)

        return [y1, y5], (kl1 + kl5)/2.0


class HGIB_v0(nn.Module):
    '''
    This function is for hypergraph information bottelneck.
    using two cascated convolution layer
    there exists kl loss calucation within the convolution layer
    '''
    def __init__(self, in_ch, n_hid=1024, n_class=3, dropout=0.5, L=2, alpha=0.5, beta=0.01, theta=0.1, use_bn = False,
        drop_rate = 0.5, heads=8):
        super(HGIB_v0, self).__init__()
        self.dropout = dropout
        self.hgc = nn.ModuleList()

        self.conv1 = HGNNPConv_GIB_v1(in_ch, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv1_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)

        # self.conv2 = HGNNPConv_GIB_v1(n_hid, n_hid*2, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv2_1 = HGNNPConv(n_hid*2, n_class, use_bn=use_bn, is_last=True)

        # self.conv3 = HGNNPConv_GIB_v1(n_hid*2, n_hid*4, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv3_1 = HGNNPConv(n_hid*4, n_class, use_bn=use_bn, is_last=True)

        # self.conv4 = HGNNPConv_GIB_v1(n_hid * 4, n_hid * 4, use_bn=use_bn, drop_rate=drop_rate, is_last=False,
        #                               heads=heads)
        # self.conv4_1 = HGNNPConv(n_hid * 4, n_class, use_bn=use_bn, is_last=True)
        # self.conv5 = HGNNPConv_GIB_v1(n_hid * 4, n_hid * 4, use_bn=use_bn, drop_rate=drop_rate, is_last=False,
        #                               heads=heads)
        # self.conv5_1 = HGNNPConv(n_hid * 4, n_class, use_bn=use_bn, is_last=True)

        # self.conv4 = HGNNPConv_GIB_v1(n_hid * 4, n_hid*2, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv4_1 = HGNNPConv(n_hid*2, n_class, use_bn=use_bn, is_last=True)

        # self.conv5 = HGNNPConv_GIB_v1(n_hid, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv5_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)


    def forward(self, x, G):
        x, kl1 = self.conv1(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        y1 = self.conv1_1(x, G)
        # x, kl2 = self.conv2(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y2 = self.conv2_1(x, G)
        # x, kl3 = self.conv3(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y3 = self.conv3_1(x, G)
        # x, kl4 = self.conv4(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y4 = self.conv4_1(x, G)
        # x, kl5 = self.conv5(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y5 = self.conv5_1(x, G)
        # x, kl6 = self.conv6(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y6 = self.conv6_1(x, G)
        # x, kl7 = self.conv7(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y7 = self.conv7_1(x, G)

        return y1, kl1



class HGIB_robust(nn.Module):
    '''
    This function is for hypergraph information bottelneck.
    using two cascated convolution layer
    there exists kl loss calucation within the convolution layer
    '''
    def __init__(self, in_ch, n_hid=1024, n_class=3, dropout=0.5, L=2, alpha=0.5, beta=0.01, theta=0.1, use_bn = False,
        drop_rate = 0.5, heads=8):
        super(HGIB_robust, self).__init__()
        self.dropout = dropout
        self.hgc = nn.ModuleList()

        self.conv1 = HGNNPConv_GIB_v1(in_ch, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv1_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)

        # self.conv2 = HGNNPConv_GIB_v1(n_hid, n_hid*2, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv2_1 = HGNNPConv(n_hid*2, n_class, use_bn=use_bn, is_last=True)

        # self.conv3 = HGNNPConv_GIB_v1(n_hid*2, n_hid*4, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv3_1 = HGNNPConv(n_hid*4, n_class, use_bn=use_bn, is_last=True)

        # self.conv4 = HGNNPConv_GIB_v1(n_hid * 4, n_hid * 4, use_bn=use_bn, drop_rate=drop_rate, is_last=False,
        #                               heads=heads)
        # self.conv4_1 = HGNNPConv(n_hid * 4, n_class, use_bn=use_bn, is_last=True)
        # self.conv5 = HGNNPConv_GIB_v1(n_hid * 4, n_hid * 4, use_bn=use_bn, drop_rate=drop_rate, is_last=False,
        #                               heads=heads)
        # self.conv5_1 = HGNNPConv(n_hid * 4, n_class, use_bn=use_bn, is_last=True)

        # self.conv4 = HGNNPConv_GIB_v1(n_hid * 4, n_hid*2, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        # self.conv4_1 = HGNNPConv(n_hid*2, n_class, use_bn=use_bn, is_last=True)

        self.conv5 = HGNNPConv_GIB_v1(n_hid, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        self.conv5_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)


    def forward(self, x, G):
        G = G.drop_hyperedges(0.2)
        x, kl1 = self.conv1(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        y1 = self.conv1_1(x, G)
        # x, kl2 = self.conv2(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y2 = self.conv2_1(x, G)
        # x, kl3 = self.conv3(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y3 = self.conv3_1(x, G)
        # x, kl4 = self.conv4(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y4 = self.conv4_1(x, G)
        x, kl5 = self.conv5(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        y5 = self.conv5_1(x, G)
        # x, kl6 = self.conv6(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y6 = self.conv6_1(x, G)
        # x, kl7 = self.conv7(x, G)
        # x = F.relu(x)
        # x = F.dropout(x, self.dropout)
        # y7 = self.conv7_1(x, G)

        return [y1, y5], (kl1 + kl5)/2.0


class HGNNP_robust2(nn.Module):
    # add feature noise
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        r = X.max(1)[0].mean(0)
        # print(r)
        e = torch.randn_like(X)
        # 0.5 is disaster 0.05 bad performance
        lam = 0.01
        noise = lam * r*e
        X += noise

        for layer in self.layers:
            X = layer(X, hg)
        return X


class HGIB_robust2(nn.Module):
    '''
    This function is for hypergraph information bottelneck.
    using two cascated convolution layer
    there exists kl loss calucation within the convolution layer
    '''
    def __init__(self, in_ch, n_hid=1024, n_class=3, dropout=0.5, L=2, alpha=0.5, beta=0.01, theta=0.1, use_bn = False,
        drop_rate = 0.5, heads=8):
        super(HGIB_robust2, self).__init__()
        self.dropout = dropout
        self.hgc = nn.ModuleList()

        self.conv1 = HGNNPConv_GIB_v1(in_ch, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads = heads)
        self.conv1_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)


        self.conv5 = HGNNPConv_GIB_v1(n_hid, n_hid, use_bn=use_bn, drop_rate=drop_rate, is_last=False, heads=heads)
        self.conv5_1 = HGNNPConv(n_hid, n_class, use_bn=use_bn, is_last=True)


    def forward(self, x, G):
        r = x.max(1)[0].mean(0)
        # print(r)
        e = torch.randn_like(x)
        # 0.5 is disaster 0.05 bad performance
        lam = 0.01
        noise = lam * r * e
        x += noise
        x, kl1 = self.conv1(x, G)
        y1 = self.conv1_1(x, G)

        x, kl5 = self.conv5(x, G)
        y5 = self.conv5_1(x, G)

        return [y1, y5], (kl1 + kl5)/2.0
