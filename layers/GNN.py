import torch
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, APPNP
import torch.nn as nn
import torch.nn.functional as F
from src.batch_whitening import IterNorm

# Borrowed from BGRL
class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm_mm=0.99, flag=1):
        super().__init__()

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.flag = flag

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.flag:  # 使用 GCNConv
                layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'), )
            else:  # 使用 MLP
                layers.append((nn.Linear(in_dim, out_dim), 'x -> x'), )
            T = 3
            #layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            layers.append(IterNorm(num_features=out_dim, num_groups=7, T=T, momentum=0.5, affine=True))
            layers.append(nn.PReLU())

        self.model_gcn = Sequential('x, edge_index', layers) if self.flag else None
        self.model_mlp = Sequential('x', layers) if not self.flag else None

    def forward(self, data):
        if self.flag:
            return self.model_gcn(data.x, data.edge_index)
        else:
            return self.model_mlp(data.x)
    def mixforward(self,x,edge_index):
        return self.model_gcn(x,edge_index)
    def reset_parameters(self):
        if self.flag:
            self.model_gcn.reset_parameters()
        else:
            self.model_mlp.reset_parameters()
