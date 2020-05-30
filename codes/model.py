# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


# ##############gcn layers#####################
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# ##############model##########################
class newModel(nn.Module):
    def __init__(self, parameters):
        super(newModel, self).__init__()
        self.input_size = parameters.loc_emb_size
        self.hidden_size = parameters.hidden_size
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_loc_graph = nn.Embedding(self.loc_size,
                                          parameters.loc_graph_emb_size)
        # self.gc2 = GraphConvolution(self.hidden_size, self.hidden_size)
        self.gc1 = GraphConvolution(parameters.loc_graph_emb_size,
                                    self.hidden_size)
        # self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.rnn_encoder = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=1,
            bidirectional=True)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'rnn_encoder.weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'rnn_encoder.weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'rnn_encoder.bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, adj, loc_train, target_len):
        h1 = torch.zeros(2, 1, self.hidden_size).cuda()
        c1 = torch.zeros(2, 1, self.hidden_size).cuda()
        loc_emb = self.emb_loc(loc)
        loc_train_emb = self.emb_loc_graph(loc_train)
        # loc_train_emb = self.dropout(loc_train_emb)
        x = loc_emb
        # x = self.dropout(x)
        loc_gcn = F.relu(self.gc1(loc_train_emb, adj))
        # loc_gcn = self.dropout(loc_gcn)
        # graph_embedding = self.fc1(loc_gcn)
        # c1_gcn = torch.max(graph_embedding, 0)[0].unsqueeze(0).unsqueeze(0)
        c1_gcn = torch.mean(loc_gcn, dim=0).unsqueeze(0).unsqueeze(0)
        # hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
        hidden_history, (h1, c1) = self.rnn_encoder(x[:-target_len], (h1, c1))
        return hidden_history, c1_gcn, (h1, c1)


class DecoderModel1(nn.Module):
    def __init__(self, parameters):
        super(DecoderModel1, self).__init__()
        self.hidden_size = parameters.hidden_size
        self.use_cuda = True
        self.rnn_type = parameters.rnn_type
        self.output_size = parameters.loc_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        input_size = self.hidden_size

        if self.rnn_type == 'GRU':
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM' or self.rnn_type == 'RNN':
            self.rnn_decoder = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=2,
                dropout=parameters.dropout_p)

        self.fc_final = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters()
              if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters()
              if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters()
             if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, h2, c2):
        loc_embedded = self.embedding(loc).view(1, 1, -1)
        if self.rnn_type == 'GRU':
            hidden_state, h2 = self.rnn_decoder(loc_embedded, h2)
        if self.rnn_type == 'LSTM':
            hidden_state, (h2, c2) = self.rnn_decoder(loc_embedded, (h2, c2))
        hidden_state = hidden_state.squeeze(1)
        # hidden_state = hidden_state.view(1, 1, 2, -1).squeeze(1).squeeze(0)
        # out = hidden_state[0].unsqueeze(0)
        out = hidden_state
        out = self.dropout(out)

        y = self.fc_final(out)
        score = F.log_softmax(y, dim=1)

        return score, (h2, c2)


# ##############Historical_model##########################
class Model(nn.Module):
    def __init__(self, parameters):
        super(Model, self).__init__()
        # data size setting
        self.loc_size = parameters.loc_size
        self.tim_size = parameters.tim_size
        self.activity_size = parameters.loc_cat_size
        # embedding setting
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_emb_size = parameters.tim_emb_size
        self.activity_emb_size = parameters.loc_cat_emb_size
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_activity = nn.Embedding(self.activity_size,
                                         self.activity_emb_size)
        # layer setting
        self.input_size = self.loc_emb_size + self.tim_emb_size + self.activity_emb_size
        self.hidden_size = parameters.hidden_size
        self.out_channels_size = self.hidden_size
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        # 1d_cnn and rnn
        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.out_channels_size,
            kernel_size=2)
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=1)
        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=2)
        self.pooling = nn.AvgPool1d(kernel_size=parameters.pool_size)
        # fc layers
        self.fc = nn.Linear(self.input_size, self.hidden_size)  # MLP
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_final = nn.Linear(self.hidden_size,
                                  self.loc_size)  # pre_add fc layer

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def forward(self, loc, tim, activity, target_len):
        # h1 = torch.zeros(2, 1, self.hidden_size).cuda()
        # c1 = torch.zeros(2, 1, self.hidden_size).cuda()
        loc_emb = self.emb_loc(loc[:-target_len])
        tim_emb = self.emb_tim(tim[:-target_len])
        activity_emb = self.emb_activity(activity[:-target_len])
        history_emb = torch.cat((loc_emb, tim_emb, activity_emb), dim=2)

        history_hidden = F.relu(self.fc(history_emb))  # MLP process
        # short-term
        # h_shot_term = self.conv(history_hidden.permute(1, 2, 0)).permute(
        #     2, 1, 0).squeeze(2)
        # h_shot_term = F.glu(h_shot_term, dim=1)
        # history_short_term, _ = torch.max(h_shot_term, dim=0)
        history_short_term, _ = torch.max(
            self.conv(history_hidden.permute(1, 2, 0)).permute(2, 1,
                                                               0).squeeze(2),
            dim=0)  # B*C*L_out->L_out*C (short-term move pattern)

        # long-term
        history_pooled = self.pooling(history_hidden.permute(1, 2, 0)).permute(
            2, 0, 1)  # B*C*L_out->L_out*B*C
        history_long_term, _ = torch.max(
            self.fc1(history_pooled.squeeze(1)), dim=0)
        # history_long_term = torch.mean(
        #     self.fc1(history_pooled.squeeze(1)), dim=0)
        history = torch.cat((history_short_term.unsqueeze(0).unsqueeze(0),
                             history_long_term.unsqueeze(0).unsqueeze(0)),
                            dim=0)
        # current trajectory
        current_loc_emb = self.emb_loc(loc[-target_len:])
        current_tim_emb = self.emb_tim(tim[-target_len:])
        current_activity_emb = self.emb_activity(activity[-target_len:])
        current_emb = torch.cat(
            (current_loc_emb, current_tim_emb, current_activity_emb), dim=2)
        pred, _ = self.gru(current_emb,
                           history_short_term.unsqueeze(0).unsqueeze(0))
        out = self.dropout(pred.squeeze(1))
        # ############################
        y = self.fc_final(out)
        score = F.log_softmax(y)

        return score