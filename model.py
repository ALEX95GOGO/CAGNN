"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data_utils import computeFFT
from cell import DCGRUCell
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

        init_hidden_state = self.encoder.init_hidden(
            batch_size).to(self._device)
        
        input_seq_att = torch.zeros(input_seq.shape)   
        x = input_seq.permute(0,2,1,3)

        se_score, input_seq_se = self.se(x.permute(0,3,2,1))
        se_score = se_score.squeeze()

        input_seq_se = input_seq_se.permute(0,2,3,1)
        x = input_seq.permute(0,3,2,1)
        x_seq = x[:,:,:].reshape(-1, 30, 10*self.input_dim)
        xa = torch.tanh(self.bnlin((x_seq-x_seq.min())/(x_seq.max()-x_seq.min())))
        
        #xa = torch.tanh(self.bnlin(x_seq)/x_seq.max())
        adj = torch.bmm(xa, xa.transpose(2, 1))  # /self.bn_features
        adj = torch.sigmoid(adj)
        adj = (adj-adj.min())/(adj.max()-adj.min())
        adj_ori_batch = adj
        adj_batch = torch.zeros(adj_mat.shape).cuda()
        
        
        for ii in range(adj_mat.shape[0]):
            A_tilde = adj[ii] + torch.eye(self.num_nodes).cuda()
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
            adj_batch[ii] = norm_adj
            #import pdb; pdb.set_trace()
        #adj_batch = adj

        supports = [adj_batch]
        input_seq_se = torch.transpose(input_seq_se, dim0=0, dim1=1)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        output_hidden, final_hidden = self.encoder(input_seq_se, init_hidden_state, supports)
        # (batch_size, max_seq_len, rnn_units*num_nodes)

        output = torch.transpose(final_hidden, dim0=0, dim1=1)
        # extract last relevant output
        last_out = utils.last_relevant_pytorch(
            output, seq_lengths, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        max_X, _ = torch.max(last_out, dim=1)
        #mean_X = torch.mean(last_out, dim=1)
        #readout = torch.cat([
        #    max_X,
        #    mean_X,
        #], dim=1)
        readout = max_X

        xdata = xdata.unsqueeze(1)
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
