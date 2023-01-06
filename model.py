"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""
from data.data_utils import computeFFT
from model.cell import DCGRUCell
from torch.autograd import Variable
import utils
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_dense_adj, get_laplacian
from torch_geometric.utils.sparse import dense_to_sparse
import scipy.io as sio
import math
import os

def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor
    """
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x)
                     for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor
    """
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(
                xs,
                dim) if isinstance(
                xs[0],
                torch.Tensor) else xs[0]) for xs in zip(
                *
                tups))
    else:
        return torch.cat(tups, dim)


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='laplacian',
                 device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)
        #self.bnlin = Linear(in_features, bn_features)  # Linear Bottleneck layer#(44*32, 32)

    def forward(self, inputs, initial_hidden_state, supports):
        #xa = torch.tanh(self.bnlin(x))
        #adj = torch.matmul(xa, xa.transpose(2, 1))  # /self.bn_features
        #adj = torch.softmax(adj, 2)
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                #_, hidden_state = self.encoding_cells[i_layer](
                #    [supports[t]], current_inputs[t, ...], hidden_state)
                _, hidden_state = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)  # (seq_len, batch_size, num_nodes * rnn_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device)  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, dcgru_activation=None,
                 filter_type='laplacian', device=None, dropout=0.0):
        super(DCGRUDecoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device
        self.dropout = dropout

        cell = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                         max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, nonlinearity=dcgru_activation,
                         filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            decoding_cells.append(cell)

        self.decoding_cells = nn.ModuleList(decoding_cells)
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)  # dropout before projection layer

    def forward(
            self,
            inputs,
            initial_hidden_state,
            supports,
            teacher_forcing_ratio=None):
        """
        Args:
            inputs: shape (seq_len, batch_size, num_nodes, output_dim)
            initial_hidden_state: the last hidden state of the encoder, shape (num_layers, batch, num_nodes * rnn_units)
            supports: list of supports from laplacian or dual_random_walk filters
            teacher_forcing_ratio: ratio for teacher forcing
        Returns:
            outputs: shape (seq_len, batch_size, num_nodes * output_dim)
        """
        seq_length, batch_size, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        go_symbol = torch.zeros(
            (batch_size,
             self.num_nodes *
             self.output_dim)).to(
            self._device)

        # tensor to store decoder outputs
        outputs = torch.zeros(
            seq_length,
            batch_size,
            self.num_nodes *
            self.output_dim).to(
            self._device)

        current_input = go_symbol  # (batch_size, num_nodes * input_dim)
        for t in range(seq_length):
            next_input_hidden_state = []
            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    supports, current_input, hidden_state)
                current_input = output
                next_input_hidden_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)

            projected = self.projection_layer(self.dropout(
                output.reshape(batch_size, self.num_nodes, -1)))
            projected = projected.reshape(
                batch_size, self.num_nodes * self.output_dim)
            outputs[t] = projected

            if teacher_forcing_ratio is not None:
                teacher_force = random.random() < teacher_forcing_ratio  # a bool value
                current_input = (inputs[t] if teacher_force else projected)
            else:
                current_input = projected

        return outputs


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix

def get_degree_matrix(adj):
    return torch.diag(sum(adj))


class GraphConvolution(nn.Module):
    """图卷积层
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层

            SAGPool中使用图卷积层计算每个图中每个节点的score

            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置

        """

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, adjacency, X):
        """图卷积层前馈

            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, input_dim], 节点特征

            Output:
            -------
            output: tensor in shape [num_nodes, output_dim], 输出

        """
        #self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        #support = torch.bmm(X, self.weight)
        #output = torch.bmm(adjacency.cuda(), support)
        support = torch.zeros([X.shape[0],X.shape[1],self.weight.shape[1]]).cuda()
        output = torch.zeros([X.shape[0], X.shape[1],self.weight.shape[1]]).cuda()
        for i in range(X.shape[0]):
            support[i] = torch.mm(X[i], self.weight)
            output[i] = torch.mm(adjacency[i].cuda(), support[i])
        if self.use_bias:
            output += self.bias

        return output

def topk(node_score, keep_ratio):
    """获取每个图的topk个重要的节点

        Inputs:
        -------
        node_score: tensor, GCN计算出的每个节点的score
        graph_batch: tensor, 指明每个节点所属的图
        keep_ratio: float, 每个图中topk的节点占所有节点的比例

        Output:
        -------
        mask: tensor, 所有节点是否是每张图中topk节点的标记

    """

    # 获得所有图的ID列表
    #graph_ids = list(set(graph_batch.cpu().numpy()))

    graph_ids = node_score.shape[0]
    # 用于记录每个图中的节点保留情况
    #mask = node_score.new_empty((0,), dtype=torch.bool)
    mask = torch.zeros([node_score.shape[0],node_score.shape[1]])

    for grid_id in range(graph_ids):
        # 遍历每张图

        # 获得此图中所有节点的score并排序
        graph_node_score = node_score[grid_id].view(-1)
        _, sorted_index = graph_node_score.sort(descending=True)

        # 获得此图图中topk节点的k数
        num_graph_node = len(graph_node_score)
        num_keep_node = int(keep_ratio * num_graph_node)
        num_keep_node = max(num_keep_node, 1)

        # 标记此图中的topk节点
        graph_mask = node_score.new_zeros((num_graph_node,), dtype=torch.bool)
        graph_mask[sorted_index[:num_keep_node]] = True

        # 拼接所有图的节点标记
        #mask = torch.cat([mask, graph_mask])
        mask[grid_id]=graph_mask


    return mask


"""
We use the batch normalization layer implemented by ourselves for this model instead using the one provided by the Pytorch library.
In this implementation, we do not use momentum and initialize the gamma and beta values in the range (-0.1,0.1). 
We have got slightly increased accuracy using our implementation of the batch normalization layer.
"""


def normalizelayer(data):
    eps = 1e-05
    a_mean = data - torch.mean(data, [0, 2, 3], True).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                             int(data.size(3)))
    b = torch.div(a_mean, torch.sqrt(torch.mean((a_mean) ** 2, [0, 2, 3], True) + eps).expand(int(data.size(0)),
                                                                                              int(data.size(1)),
                                                                                              int(data.size(2)),
                                                                                              int(data.size(3))))

    return b


class Batchlayer(torch.nn.Module):
    def __init__(self, dim):
        super(Batchlayer, self).__init__()
        self.gamma = torch.nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.beta = torch.nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.gamma.data.uniform_(-0.1, 0.1)
        self.beta.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        data = normalizelayer(input)
        gammamatrix = self.gamma.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        betamatrix = self.beta.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))

        return data * gammamatrix + betamatrix

'''
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn
'''        
########## Model for seizure classification/detection ##########
class DCRNNModel_classification(nn.Module):
    def __init__(self, args, num_classes, aptinit=None,device=None, training=True):
        super(DCRNNModel_classification, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        max_diffusion_step = args.max_diffusion_step
        self.input_dim = args.input_dim
        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type)

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(rnn_units, rnn_units//2)
        self.fc2 = nn.Linear(rnn_units//2, rnn_units//4)
        self.fc3 = nn.Linear(rnn_units//4, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.act = nn.Tanh()
        self.gcn = GraphConvolution(rnn_units, 1)
        self.act1 = nn.Tanh()
        self.gcn1 = GraphConvolution(rnn_units, 1)
        self.keep_ratio=float(args.keep_ratio)
        self.input_dim = args.input_dim

        # 输出层
        self.mlp = nn.Sequential(
            nn.Linear(rnn_units*2, rnn_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(rnn_units, rnn_units // 2),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_units // 2, num_classes),
        )

        self.feature = 30
        self.padding = torch.nn.ReplicationPad2d((31, 32, 0, 0))
        self.conv = torch.nn.Conv2d(self.feature, self.feature, (1, 64),
                                    groups=self.feature)  # ,padding=(0,32),padding_mode='replicate')
        self.batch = Batchlayer(self.feature)
        self.avgpool = torch.nn.AvgPool2d((1, 64))
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.lstm = torch.nn.LSTM(30, 2)

        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.lstm = torch.nn.LSTM(30, 2)
        self.se = SE_Block(enc_input_dim, 2)

        self.conv1 = nn.Conv2d(1, 1, (5, 5))
        self.channels = 30
        node_to_keep = args.node_to_keep
        self.topk = node_to_keep

        self.bnlin = Linear(enc_input_dim*10, 3)  # Linear Bottleneck layer#(44*32, 32)
        
        self.w1 = Linear(30, 5)
        self.w2 = Linear(30, 5)
        self.w3 = Linear(30, 5)
        
        self.training = training
        kernelLength=64
        sampleLength=640
        N1=16
        d=2
        self.conv = torch.nn.Conv2d(self.channels,self.channels,(1,kernelLength),groups=30)
        self.batch = torch.nn.BatchNorm2d(self.channels)
        self.GAP = torch.nn.AvgPool2d((1,sampleLength-kernelLength+1))
        #self.bnlin2 = Linear(30, 2)  # Linear Bottleneck layer#(44*32, 32)
        #self.bnlin3 = Linear(30, 2)  # Linear Bottleneck layer#(44*32, 32)
        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        #self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes

        #if self.edge_additions:
        #    self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        #else:
        #    self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.pointwise = torch.nn.Conv2d(1,N1,(self.channels,1))
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1) 
        self.activ=torch.nn.ReLU()       
        self.batchnorm = torch.nn.BatchNorm2d(d*N1, track_running_stats=True)     

    def _compute_supports(self, adj_mat):
        """
        Comput supports
        """
        supports = []
        supports_mat = []
        self.filter_type = "dual_random_walk"
        if self.filter_type == "laplacian":  # ChebNet graph conv
            supports_mat.append(
                utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif self.filter_type == "random_walk":  # Forward random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
        elif self.filter_type == "dual_random_walk":  # Bidirectional random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
            supports_mat.append(
                utils.calculate_random_walk_matrix(adj_mat.T).T)
        else:
            supports_mat.append(utils.calculate_scaled_laplacian(adj_mat))
        for support in supports_mat:
            supports.append(torch.FloatTensor(support.toarray()))
        return supports

    def forward(self, xdata, input_seq, seq_lengths, supports,adj_mat,block_chan=None):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # (max_seq_len, batch, num_nodes, input_dim)

        # initialize the hidden state of the encoder
        #input_seq = torch.cat([input_seq[:,:,0:7], input_seq[:,:,8:16], input_seq[:,:,17:]], dim=2)
        init_hidden_state = self.encoder.init_hidden(
            batch_size).to(self._device)
        
        input_seq_att = torch.zeros(input_seq.shape)   
        x = input_seq.permute(0,2,1,3)

        se_score, input_seq_se = self.se(x.permute(0,3,2,1))
        se_score = se_score.squeeze()
        #import pdb; pdb.set_trace()
        '''
        if self.training == False:
            print(se_score.shape)
            if os.path.exists("se_score_32.mat"):
              se_score_cat = sio.loadmat("se_score_32.mat")["se_score"]
              se_score_cat = torch.cat((torch.from_numpy(se_score_cat).cuda(),se_score),0)
            else:
              se_score_cat = se_score
            sio.savemat("se_score_32.mat",{"se_score": se_score_cat.cpu().detach().numpy()})
        '''
        #print(torch.mean(se_score,0))
        
        input_seq_se = input_seq_se.permute(0,2,3,1)
 
        #x_in = torch.zeros(input_seq.permute(0,3,1,2).shape).cuda()
        #for seq in range(6):
        
        #q = self.w1(input_seq.permute(0,3,1,2))
        #k = self.w2(input_seq.permute(0,3,1,2))
        #v = self.w3(input_seq.permute(0,3,1,2))
        #import pdb; pdb.set_trace()
        #multihead_attn = nn.MultiheadAttention(10, 2, batch_first=True).cuda()
        #attn_output, attn_output_weights = multihead_attn(q[:,:,0], k[:,:,0], v[:,:,0])
        #for ii in range(6):
        
        #q = q.permute(0,2,1,3).reshape(batch_size*6,3,5)
        #k = k.permute(0,2,1,3).reshape(batch_size*6,3,5)
        #att_score = torch.softmax(torch.bmm(q, k.permute(0,2,1)),2)
        #print(att_score[0,0])
        #x_in = torch.matmul(att_score,input_seq.reshape(batch_size*6,30,3).permute(0,2,1))
        #x_in = x_in.reshape(batch_size,6,30,3)
        x = input_seq.permute(0,3,2,1)
        #x_seq = x_in.permute(0,2,1)
        x_seq = x[:,:,:].reshape(-1, 30, 10*self.input_dim)
        #xa = self.bnlin(x_seq)
        #adj_6=torch.zeros(6,adj_mat.shape[0],30,30).cuda()
        #for ii in range(6):
        #    x_seq = x[:,:,ii]
        #xa = self.bnlin(x_seq/x_seq.max())
        #xa = self.bnlin((x_seq-x_seq.min())/(x_seq.max()-x_seq.min()))
        xa = torch.tanh(self.bnlin((x_seq-x_seq.min())/(x_seq.max()-x_seq.min())))
        
        #xa = torch.tanh(self.bnlin(x_seq)/x_seq.max())
        adj = torch.bmm(xa, xa.transpose(2, 1))  # /self.bn_features
        adj = torch.sigmoid(adj)
        adj = (adj-adj.min())/(adj.max()-adj.min())
        #import pdb; pdb.set_trace()
        #adj = (adj-torch.min(adj,0))/(torch.max(adj,0)-torch.min(adj,0))
        
        '''
        if self.training == False:
            #print(se_score.shape)
            if os.path.exists("adj_score.mat"):
              adj_cat = sio.loadmat("adj_score.mat")["adj_score"]
              adj_cat = torch.cat((torch.from_numpy(adj_cat).cuda(),adj),0)
            else:
              adj_cat = adj
            sio.savemat("adj_score.mat",{"adj_score": adj_cat.cpu().detach().numpy()})
        '''
        
        #import pdb; pdb.set_trace()
        #adj = torch.tanh(adj)
        #adj = torch.nn.ReLU()(adj)
        #print(adj)
        #adj_6[ii]=adj
        #adj = torch.softmax(adj, 2)

        #adj = adj_mat
        #amask = torch.ones(xa.size(0), self.channels, self.channels).cuda()
        #amask.fill_(0.0)
        #s, t = adj.topk(self.topk, 2)
        #import pdb; pdb.set_trace()
        #amask.scatter_(2, t, s.fill_(1))

        #score_mat = sio.loadmat('adj_score_sub1.mat')['adj_score']
        #rank_score_mat = np.sum(np.sum(score_mat,axis=0),axis=0)
        #order = rank_score_mat.argsort()
        #order = 29-order
        #import pdb; pdb.set_trace()
        #if block_chan != None:
        #    for ii in range(1,block_chan+1):
        #        amask[:,order[ii]]=0
        #        amask[order[ii],:]=0
        
        #adj = adj * amask
        #graph_loss += sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        adj_ori_batch = adj
        #import pdb; pdb.set_trace()
        #for k in range(adj_mat.shape[0]):
        #    adj_ori_batch[k] = adj
        #self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec.cpu(), self.num_nodes)  # Ensure symmetry
        #print(adj)
        adj_batch = torch.zeros(adj_mat.shape).cuda()
        #adj_batch = torch.zeros(6,adj_mat.shape[0],30,30).cuda()

        #adj_i = torch.zeros(adj_mat.shape).cuda()
        #for k in range(adj_mat.shape[0]):
        #    adj_i[k] = adj
        #import pdb; pdb.set_trace()
        #for aa in range(6):
        
        
        for ii in range(adj_mat.shape[0]):
            A_tilde = adj[ii] + torch.eye(self.num_nodes).cuda()
            #A_tilde[A_tilde>1]=1
            #A_tilde = adj[ii]
            #A_tilde = torch.eye(self.num_nodes)
            D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this
            # Raise to power -1/2, set all infs to 0s
            D_tilde_exp = D_tilde ** (-1 / 2)
            D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
            d_inv = torch.linalg.inv(D_tilde_exp).flatten()
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diag(d_inv)

            # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
            norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp).cuda()
            sum_adj = sum(norm_adj)
            #order = sum_adj.argsort()
            #rank = order.argsort()
            #for k in range(self.topk):
            #    norm_adj[rank==k,:]=0
            #    norm_adj[:,rank==k]=0
            adj_batch[ii] = norm_adj
            #import pdb; pdb.set_trace()
        #amask = torch.ones(xa.size(0), self.channels, self.channels).cuda()
        #amask.fill_(0.0)
        #s, t = adj_batch.topk(self.topk, 2)
        #amask.scatter_(2, t, s.fill_(1))
        #adj_batch = adj_batch * amask

        #adj_batch = adj

        supports = [adj_batch]
        #supports.append(adj_batch)
        #supports = adj_batch
        #    supports.append(adj_batch)
        input_seq_se = torch.transpose(input_seq_se, dim0=0, dim1=1)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        #x_in = torch.transpose(input_seq, dim0=0, dim1=1)
        #input_seq_att = torch.transpose(input_seq_att, dim0=0, dim1=1)
        #input_seq_att = input_seq_att.cuda()
        #output_hidden, final_hidden = self.encoder(x_in, init_hidden_state, supports)
        output_hidden, final_hidden = self.encoder(input_seq_se, init_hidden_state, supports)
        # (batch_size, max_seq_len, rnn_units*num_nodes)

        output = torch.transpose(final_hidden, dim0=0, dim1=1)
        # extract last relevant output
        last_out = utils.last_relevant_pytorch(
            output, seq_lengths, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)

        '''
        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            output_hidden,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio)  # (seq_len, batch_size, num_nodes * output_dim)


        
        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)
        #import pdb; pdb.set_trace()
        '''
        #layer1_output = output_hidden[0].view(batch_size, self.num_nodes, self.rnn_units)
        #adj_copy = adj.clone()
        #node_score1 = self.gcn1(adj, layer1_output.to(self._device))
        # print(torch.mean(node_score,0))
        #node_score1 = self.act1(node_score1)
        #mask1 = topk(node_score1, self.keep_ratio)

        #last_out1 = layer1_output * torch.unsqueeze(mask1, 2).cuda()
        #mask_X1 = last_out1 * node_score1
        #max_X1, _ = torch.max(last_out, dim=1)
        #mean_X1 = torch.mean(last_out, dim=1)
        #readout1 = torch.cat([
        #    max_X1,
        #    mean_X1,
        #], dim=1)

        #adj_copy = adj.clone()
        #node_score = self.gcn(adj_copy, last_out)
        #node_score = F.softmax(node_score)
        #node_score = self.act(node_score)
        #print(node_score[0])
        #import pdb; pdb.set_trace()
        #mask = topk(node_score, self.keep_ratio)

        #last_out = last_out * torch.unsqueeze(mask, 2).cuda()
        #print(torch.where(mask[0] == 1))
        #print(node_score[0])
        #mask_X = last_out * node_score

        # final FC layer
        #import pdb; pdb.set_trace()
        
        max_X, _ = torch.max(last_out, dim=1)
        mean_X = torch.mean(last_out, dim=1)
        readout = torch.cat([
            max_X,
            mean_X,
        ], dim=1)

        '''
        if self.training == False:
            print(readout.shape)
            if os.path.exists("readout.mat"):
              readout_cat = sio.loadmat("readout.mat")["readout"]
              readout_cat = torch.cat((torch.from_numpy(readout_cat).cuda(),readout),0)
            else:
              readout_cat = readout
            sio.savemat("readout.mat",{"readout": readout_cat.cpu().detach().numpy()})
            readout = readout.to(self._device)
        '''
        xdata = xdata.unsqueeze(1)
        '''
        intermediate = self.conv(xdata)
        intermediate = self.batch(intermediate)
        #intermediate = self.batchnorm(intermediate)
        intermediate = torch.nn.ELU()(intermediate)    
        intermediate = self.GAP(intermediate)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        
        intermediate = self.pointwise(xdata)    
        intermediate = self.depthwise(intermediate)    
        intermediate = self.activ(intermediate) 
        intermediate = self.batchnorm(intermediate)
        intermediate = self.GAP(intermediate)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        '''
        #import pdb; pdb.set_trace()
        #readout = readout+readout1
        # logits = self.fc3(self.relu(self.dropout(self.fc2(self.relu(self.dropout(self.fc1(self.relu(self.dropout(mask_X)))))))))
        #readout = torch.cat((readout,intermediate),1)
        pool_logits = self.mlp(readout)
        #pool_logits = nn.LogSoftmax(dim=1)(pool_logits)

        return pool_logits, adj
########## Model for seizure classification/detection ##########

class SE_Block(torch.nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=5):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(c, c // r, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(c // r, c, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        #print(torch.mean(y,0).squeeze())
        return y, x * y.expand_as(x)


    
