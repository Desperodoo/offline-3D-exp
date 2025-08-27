import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import (
    SGConv,
    GCNConv,
    GATConv,
    SAGEConv,
    TransformerConv,
    APPNP,
    JumpingKnowledge,
    MessagePassing,
)

# ------------------------------ Refactored Code ------------------------------ #

class MPNNs(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        in_dropout=0.0,
        dropout=0.0,
        heads=1,
        pre_ln=True,
        bn=False,
        res=True,
        jk=True,
        conv="GCN",
    ):
        """
        Message Passing Neural Networks (MPNNs) for graph-based drl.

        This class implements a multi-layer gnn with various types of graph convolution layers.
        It supports different configurations such as pre-layer normalization, batch normalization, residual connections,
        and jumping knowledge connections.

        reference:
        https://github.com/LUOyk1999/tunedGNN/blob/main/medium_graph/model.py
        https://zhuanlan.zhihu.com/p/345353294
        """
        super(MPNNs, self).__init__()
        assert bn is False, "Batch Norm isn't suitable for RL."
        assert dropout <= 0.2 and in_dropout <= 0.2, "Dropout rate should be less than 0.2."

        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.bn = bn

        self.res = res
        self.jk = jk

        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()

        ## first layer
        if conv == "TRANS":
            self.local_convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, concat=False))
        elif conv == "GAT":
            self.local_convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        elif conv == "SAGE":
            self.local_convs.append(SAGEConv(in_channels, hidden_channels))
        else:
            self.local_convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        if self.pre_ln:
            self.pre_lns.append(nn.LayerNorm(in_channels))
        if self.bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        ## following layers
        for _ in range(local_layers - 1):
            if conv == "TRANS":
                self.local_convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, concat=False))
            elif conv == "GAT":
                self.local_convs.append(GATConv(hidden_channels, hidden_channels, heads=heads))
            elif conv == "SAGE":
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(
                    GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True)
                )

            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(nn.LayerNorm(hidden_channels))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.pred_local = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.bn:
            for p_bn in self.bns:
                p_bn.reset_parameters()
        self.pred_local.reset_parameters()

    def forward(self, data):
        """
        Forward pass of the MPNNs model.

        Args:
            data (torch_geometric.data.Data): Input data object containing graph information.
                - data.x (torch.Tensor): Node features of shape (num_nodes, in_channels).
                - data.edge_index (torch_sparse.SparseTensor): Sparse tensor of shape (num_nodes, num_nodes).
                - data.edge_weight (torch.Tensor, optional): Edge weights of shape (num_edges,).

        Returns:
            torch.Tensor: Output tensor after applying graph convolution layers.
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
        x_final = 0

        x = F.dropout(x, p=self.in_drop, training=self.training)

        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            if edge_weight is not None:
                if self.res:
                    x = local_conv(x, edge_index, edge_weight) + self.lins[i](x)
                else:
                    x = local_conv(x, edge_index, edge_weight)
            else:
                if self.res:
                    x = local_conv(x, edge_index) + self.lins[i](x)
                else:
                    x = local_conv(x, edge_index)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x

        x = self.pred_local(x_final)

        return x

# ------------------------------ Refactored Code ------------------------------ #

class GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.0,
        save_mem=True,
        use_bn=False,
        use_ln=True,
        use_residual=True,
        use_jk=True,
    ):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=not save_mem))
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, cached=not save_mem))  # to be fixed
        self.pred_local = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.residual_transform = None
        self.residual = use_residual
        self.use_jk = use_jk

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, data):
        """
        Forward pass of the GCN model.

        Args:
            data (torch_geometric.data.Data): Input data object containing graph information.
                - data.x (torch.Tensor): Node features of shape (num_nodes, in_channels).
                - data.edge_index (torch_sparse.SparseTensor): Sparse tensor of shape (num_nodes, num_nodes).
                - data.edge_weight (torch.Tensor, optional): Edge weights of shape (num_edges,).

        Returns:
            torch.Tensor: Output tensor after applying graph convolution layers.
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
        x_final = 0

        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            if self.use_ln:
                x = self.lns[i](x)
            if edge_weight is not None:
                if self.residual:
                    x = conv(x, edge_index, edge_weight) + self.lins[i](x)
                else:
                    x = conv(x, edge_index, edge_weight)

            else:
                if self.residual:
                    x = conv(x, edge_index) + self.lins[i](x)
                else:
                    x = conv(x, edge_index)

            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_jk:
                x_final = x_final + x
            else:
                x_final = x

        x = self.pred_local(x_final)
        # Apply the final GCN layer
        # if edge_weight is not None:
        #     x = self.convs[-1](x, edge_index, edge_weight)
        # else:
        #     x = self.convs[-1](x, edge_index)

        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.0,
        use_bn=False,
        use_ln=True,
        use_residual=True,
        use_jk=True,
        heads=8,
        out_heads=1,
    ):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        # self.convs.append(GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=True))

        # self.bns = nn.ModuleList()
        # self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        # for _ in range(num_layers - 2):
        #     self.convs.append(
        #         GATConv(hidden_channels * heads, hidden_channels, dropout=dropout, heads=heads, concat=True)
        #     )
        #     self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # self.convs.append(
        #     GATConv(hidden_channels * heads, out_channels, dropout=dropout, heads=out_heads, concat=False)
        # )

        self.convs.append(GATConv(in_channels, hidden_channels, dropout=dropout, heads=heads, concat=False))
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.lns = nn.ModuleList()
        # self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if use_ln:
            self.lns.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, dropout=dropout, heads=heads, concat=False)
            )
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            # self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(nn.LayerNorm(hidden_channels))
        # self.convs.append(
        #     GATConv(hidden_channels, out_channels, dropout=dropout, heads=out_heads, concat=False)
        # )
        self.pred_local = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout
        # self.activation = F.elu
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_ln = use_ln
        # self.residual_transform = None
        self.residual = use_residual
        self.use_jk = use_jk

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, data):
        x = data.x
        x_final = 0

        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            if self.use_ln:
                x = self.lns[i](x)

            # x = conv(x, data.edge_index)

            # Add residual connection
            if self.residual:
                x = conv(x, data.edge_index) + self.lins[i](x)
            else:
                x = conv(x, data.edge_index)

            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_jk:
                x_final = x_final + x
            else:
                x_final = x

        # x = self.convs[-1](x, data.edge_index)
        x = self.pred_local(x_final)
        return x


def full_attention_conv(qs, ks, vs, output_attn=False):
    """
    Compute full attention convolution.

    Args:
        qs (torch.Tensor): Query tensor [N, H, M].
        ks (torch.Tensor): Key tensor [L, H, M].
        vs (torch.Tensor): Value tensor [L, H, D].
        output_attn (bool, optional): Return attention weights if True.

    Returns:
        torch.Tensor: Attention output [N, H, D].
        torch.Tensor (optional): Attention weights [N, N] if output_attn is True.
    """
    # normalize input 对QK进行归一化
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs  # add self-loop

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    # 以上过程不会得到显式attention，需要可视化再算出来
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    """
    transformer with fast attention
    """

    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)  # [N, D]

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        alpha=0.5,
        dropout=0.0,
        use_bn=False,
        use_residual=True,
        use_weight=True,
        use_act=False,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight)
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data.x
        # 边的信息在transconv中实际没有用到
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None

        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        num_heads=1,
        alpha=0.5,
        dropout=0.0,
        use_bn=False,
        use_residual=True,
        use_weight=True,
        use_graph=True,
        use_act=False,
        graph_weight=0.8,
        gnn=None,
        aggregate="add",
    ):
        super().__init__()
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            num_layers,
            num_heads,
            alpha,
            dropout,
            use_bn,
            use_residual,
            use_weight,
        )
        self.gnn = gnn  # gnn从外部传入
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act

        self.aggregate = aggregate
        # gnn和transformer输出的聚合方式
        if aggregate == "add":
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == "cat":
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f"Invalid aggregate type:{aggregate}")

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, data):
        """
        Forward pass of the SGFormer model.

        Args:
            data (torch_geometric.data.Data): Input data object containing graph information.
            - data.x (torch.Tensor): Node features of shape (num_nodes, in_channels).
            - data.edge_index (torch_sparse.SparseTensor): Sparse tensor of shape (num_nodes, num_nodes).
            - data.edge_weight (torch.Tensor, optional): Edge weights of shape (num_edges,).

        Returns:
            torch.Tensor: Output tensor after applying transformer convolution, optional graph neural network,
                  and fully connected layer. The shape of the output tensor is (num_nodes, out_channels).
        """
        x1 = self.trans_conv(data)
        if self.use_graph:
            x2 = self.gnn(data)
            if self.aggregate == "add":
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()


# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)   # n_batch*n_query,n_dim

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)    # n_batch*n_query,n_dim × n_dim,key_dim 
        K = torch.matmul(k_flat, self.w_key).view(shape_k)      # n_batch*n_key,n_dim × n_dim,key_dim
        
        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))   # n_batch,n_query,key_dim × n_batch,n_key,key_dim
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention

# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        """
        Perform the forward pass of the multi-head attention mechanism.
        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, n_query, embedding_dim).
            k (torch.Tensor, optional): Key tensor of shape (batch_size, n_key, embedding_dim). Defaults to None.
            v (torch.Tensor, optional): Value tensor of shape (batch_size, n_value, embedding_dim). Defaults to None.
            key_padding_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, n_key) indicating padding positions. Defaults to None.
            attn_mask (torch.Tensor, optional): Mask tensor of shape (batch, n_query, n_key) indicating positions not to attend to. Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_query, embedding_dim).
            torch.Tensor: Attention weights of shape (n_heads, batch_size, n_query, n_key).
        """
        
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.reshape(-1, input.size(-1))).reshape(*input.size())


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w

