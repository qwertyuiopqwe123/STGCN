import copy
import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose
from src.utils import  feature_drop_weights, drop_feature_weighted_2,  pseudo_drop_weights,cal_Weights
import scipy.sparse as sp
import numpy as np

class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]

def other_transform(args,data,device, predict_lbl_pro, degree_sim,  weights, Z=None):
        drop_feature_rate_1=args.drop_feature_rate_1
        drop_edge_rate_1=args.drop_edge_rate_1
        node_w_ps,new_index, new_prediction=cal_Weights(Z, predict_lbl_pro, degree_sim, device, weights)
        node_w_ps=node_w_ps.to(device)
        drop_weights = pseudo_drop_weights(data, node_w_ps).to(device)
        node_pseduo = degree_sim.float().to(device)
        feature_weights = feature_drop_weights(data.x, node_c=node_pseduo).to(device)
        edge_index1 = drop_edge_weighted(data.edge_index, drop_weights, drop_edge_rate_1, threshold=0.7)

        x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_1)
        return edge_index1, x_1,new_index, new_prediction

def gdc(A: sp.csr_matrix, alpha=0.5, eps=0.0001):
    N = A.shape[0]
    A_loop = sp.eye(N, format='csr') + A  # 确保是 CSR 格式
    D_loop_vec = A_loop.sum(0).A1  # 获取度数向量
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)  # 创建对角矩阵
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt  # 计算对称传播矩阵
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)  # 计算传播矩阵
    S_tilde = S.multiply(S >= eps)  # 去除小值
    D_tilde_vec = S_tilde.sum(0).A1  # 获取新的度数向量
    T_S = S_tilde / D_tilde_vec  # 归一化
    return sp.csr_matrix(T_S)

def create_diffusion_graph(A: sp.csr_matrix, x: torch.Tensor, device,alpha=0.5, eps=0.0001):
    # 保持 A 在 GPU 上进行稀疏运算，避免转换为 NumPy 数组
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()  # 在将 A 转换为稀疏矩阵时，将其转移到 CPU
    A_csr = sp.csr_matrix(A)  # 转换为 CSR 格式稀疏矩阵
    # 使用 gdc 计算扩散矩阵（假设 gdc 返回稀疏矩阵）
    T_S = gdc(A_csr, alpha=alpha, eps=eps)
    # 保持 T_S 为稀疏矩阵，避免转为稠密矩阵
    coo = T_S.tocoo()  # 保持稀疏表示
    # 转换为 PyTorch 的稀疏张量
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    T_S_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))
    # 计算更新后的特征，使用稀疏矩阵乘法
    updated_features = torch.sparse.mm(T_S_tensor, x)
    # 创建边索引
    edge_index = torch.tensor(np.vstack((T_S.nonzero())), dtype=torch.long, device=device)
    return edge_index, updated_features


