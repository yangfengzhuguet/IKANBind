########################################################################################################################
#
# Hypergraphs are used to model interactions between multiple residues.
# Hypergraph convolution and hypergraph attention convolution are used to extract latent discriminative representations.
# Gated multi-head attention mechanism is used to fuse the outputs of hypergraph hypergraph convolution and hypergraph attention convolution.
# KAN, with explicit mathematical expressions, are used to describe the mapping between input features and output probabilities during classification.
#
########################################################################################################################

from torch import nn
from torch.nn.parameter import Parameter
from SymbolicKANLinear import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Gated self-attention mechanism
class Attention_1(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(Attention_1, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        self.sigmoid = torch.nn.Sigmoid()
        self.hidden_size = hidden_size

        # self.layer_norm = torch.nn.LayerNorm(hidden_size).to(device)

        self.query = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.key = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.value = nn.Linear(hidden_size, self.hidden_size, bias=False).to(device)
        self.gate = nn.Linear(hidden_size, self.hidden_size).to(device)

    def transpose_for_scores(self, x):  # Divide the vector into two heads
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)  # [:-1]Left closed and right open not included-1
        x = x.view(*new_x_shape)  # * The purpose of the asterisk is probably to remove the tuple attribute (automatic unpacking)
        return x.permute(0, 2, 1, 3)

    def forward(self, batch_hidden):
        query = self.query(batch_hidden)
        key = self.key(batch_hidden)
        value = self.key(batch_hidden)
        gate = self.sigmoid(self.gate(batch_hidden))
        query = self.transpose_for_scores(query)  # batch,num_attention_heads,len,attention_head_size

        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        outputs = torch.matmul(key, query.transpose(-1, -2))  # b x num_attention_heads*len*len

        attention_scores = outputs / math.sqrt(self.attention_head_size)  # (batch,num_attention_heads,len,len)
        attn_scores = F.softmax(attention_scores, dim=-1)  #

        # sum weighted sources
        context_layer = torch.matmul(attn_scores, value)  # (batch,num_attention_heads,len,attention_head_size

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # (batch,n,num_attention_heads,attention_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size, 1)
        batch_outputs = context_layer.view(*new_context_layer_shape)  # (batch,n,all_head_size)
        batch_outputs = gate * batch_outputs.squeeze(3)

        batch_outputs = batch_outputs

        return batch_outputs, attn_scores
class EfficientHypergraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.2, dropout=0.2):
        super().__init__()
        self.P = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout

    def forward(self, X, H, W_e=None):
        eps = 1e-9
        N, E = H.shape
        device = X.device

        Z = self.P(X)  # [N, F]

        # -------- Hyperedge feature --------
        edge_deg = H.sum(dim=0) + eps
        edge_feat = (H.t() @ Z) / edge_deg.unsqueeze(1)  # [E, F]

        # -------- Sparse attention --------
        idx_i, idx_e = H.nonzero(as_tuple=True)
        zi = Z[idx_i]
        he = edge_feat[idx_e]

        e = self.leakyrelu(
            torch.cat([zi, he], dim=1) @ self.a
        ).squeeze()

        H_c = torch.zeros_like(H)
        H_c[idx_i, idx_e] = e

        H_c = F.softmax(H_c, dim=1)
        H_c = F.dropout(H_c, self.dropout, training=self.training)

        if W_e is None:
            W_e = torch.ones(E, device=device)

        D = H_c @ W_e + eps
        B = H_c.sum(dim=0) + eps

        D_inv_sqrt = D.pow(-0.5)
        B_inv = B.pow(-1.0)

        # -------- Node → Hyperedge --------
        Z1 = (H_c.t() * B_inv.unsqueeze(1)) @ (D_inv_sqrt.unsqueeze(1) * Z)

        # -------- Hyperedge → Node --------
        X_next = (H_c * W_e.unsqueeze(0)) @ Z1
        X_next = D_inv_sqrt.unsqueeze(1) * X_next

        return F.elu(X_next)

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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.LayerNorm = torch.nn.LayerNorm(256)
        self.linear1 = nn.Linear(256, 128, bias=True)
        self.linear2 = nn.Linear(128, 64, bias=True)
        self.linear3 = nn.Linear(64, 1, bias=True)
        self.linear4 = nn.Linear(256, 1, bias=True)

    def forward(self, feat):
        # feat = self.LayerNorm(feat)
        pair_feat2 = F.relu(self.linear1(feat))
        pair_feat3 = F.relu(self.linear2(pair_feat2))
        pair_feat4 = self.linear3(pair_feat3)
        # pair_feat4 = self.linear4(feat)
        return torch.sigmoid(pair_feat4)

class KAN(nn.Module): # custom function
    def __init__(self):
        super(KAN, self).__init__()
        self.kanlayer1 = KANLinear(256, 128, num_basis=10) # 10
        self.kanlayer2 = KANLinear(128, 64, num_basis=10) # 10
        self.kanlayer3 = KANLinear(64, 1, num_basis=10) # 10
        self.LayerNorm = torch.nn.LayerNorm(256)

    def forward(self, feat):
        pair_feat2 = self.kanlayer1(feat)
        pair_feat3 = self.kanlayer2(pair_feat2)
        pair_feat4 = self.kanlayer3(pair_feat3)
        return torch.sigmoid(pair_feat4), pair_feat3

class HGNN(nn.Module):
    def __init__(self, n_feat, hidden_dim, out_dim):
        super(HGNN, self).__init__()
        self.hgcn1 = HGNN_conv(n_feat, hidden_dim)
        self.hgcn2 = HGNN_conv(hidden_dim, out_dim)
        self.hgac1 = EfficientHypergraphAttention(n_feat, hidden_dim)
        self.hgac2 = EfficientHypergraphAttention(hidden_dim, 128)
        self.multihead_attention = nn.ModuleList([Attention_1(hidden_size=256 + 128, num_attention_heads=16) for _ in range(8)])
        self.res = nn.Linear(n_feat, out_dim)
        self.KAN = KAN()
        self.MLP = MLP()
        self.fc1 = nn.Linear((256 + 128), 256)

    def forward(self, X, G, H):
        x_residual = X
        x_residual = F.relu(self.res(x_residual))

        feat_update_1 = F.relu(self.hgcn1(X, G))
        feat_update_1 = F.dropout(feat_update_1, 0.3)
        feat_update_11 = F.relu(self.hgcn2(feat_update_1, G))
        feat_update_2 = self.hgac1(X, H)
        feat_update_22 = self.hgac2(feat_update_2, H)

        feat_update_11 = torch.unsqueeze(feat_update_11, dim=0)
        feat_update_22 = torch.unsqueeze(feat_update_22, dim=0)

        fea = torch.cat([feat_update_11, feat_update_22], dim=2)


        embeddings = F.relu(self.fc1(fea))
        embeddings = embeddings.squeeze(0)
        embeddings = x_residual + embeddings


        pred = self.KAN(embeddings)

        return pred


