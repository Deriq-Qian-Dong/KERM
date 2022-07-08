#-*- coding: utf-8 -*-
import numpy as np
import math

import paddle
import paddle as P
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

import pgl
import pgl.nn as gnn
from pgl.nn import functional as GF
from pgl.utils.logger import log

def _build_linear(n_in, n_out, name, init):
    return nn.Linear(
        n_in,
        n_out,
        weight_attr=P.ParamAttr(
            name='%s.w_0' % name if name is not None else None,
            initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None, )

def batch_norm_1d(num_channels):
    if dist.get_world_size() > 1:
        return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm1D(num_channels))
    else:
        return nn.BatchNorm1D(num_channels)

class LiteGEMConv(paddle.nn.Layer):
    def __init__(self, config, with_efeat=True):
        super(LiteGEMConv, self).__init__()
        log.info("layer_type is %s" % self.__class__.__name__)
        self.config = config
        self.with_efeat = with_efeat
        self.aggr = self.config['aggr']
        self.eps = 1e-7
        self.emb_dim = self.config['gnn_hidden_size']
        initializer = nn.initializer.TruncatedNormal(
            std=config['initializer_range'])
        self.f1 = _build_linear(config['gnn_hidden_size']*2, config['gnn_hidden_size'], name=None, init=initializer)
        self.f2 = _build_linear(config['gnn_hidden_size']*2, config['gnn_hidden_size'], name=None, init=initializer)
        self.f3 = _build_linear(config['gnn_hidden_size']*2, config['gnn_hidden_size'], name=None, init=initializer)
        self.fc_concat = Linear(self.emb_dim * 3, self.emb_dim)
        assert self.aggr in ['softmax_sg', 'softmax', 'power']

        channels_list = [self.emb_dim]
        for i in range(1, self.config['mlp_layers']):
            channels_list.append(self.emb_dim * 2)
        channels_list.append(self.emb_dim)

        self.mlp = MLP(channels_list,
                       norm=self.config['norm'],
                       last_lin=True)

    def send_func(self, src_feat, dst_feat, edge_feat):
        # h = paddle.concat([dst_feat['h'], src_feat['h'], edge_feat['e']], axis=1)
        # h = self.fc_concat(h)
        # 源节点到边的转移概率，边到目标节点的转移概率
        src_feat = src_feat['h']
        dst_feat = dst_feat['h']
        edge_feat = edge_feat['e']
        h = self.f1(paddle.concat([src_feat, edge_feat],axis=1))+self.f2(paddle.concat([edge_feat, dst_feat],axis=1))+self.f3(paddle.concat([src_feat, dst_feat],axis=1))
        msg = {"h": F.swish(h) + self.eps}
        return msg

    def recv_func(self, msg):
        alpha = msg.reduce_softmax(msg["h"])
        out = msg['h'] * alpha
        out = msg.reduce_sum(out)
        return out


    def forward(self, graph, nfeat, efeat=None):
        msg = graph.send(src_feat={"h": nfeat},
                            dst_feat={"h": nfeat},
                            edge_feat={"e": efeat},
                            message_func=self.send_func)
        out = graph.recv(msg=msg, reduce_func=self.recv_func)
        out = nfeat + out
        out = self.mlp(out)
        return out




def Linear(input_size, hidden_size, with_bias=True):
    fan_in = input_size
    bias_bound = 1.0 / math.sqrt(fan_in)
    fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
	low=-bias_bound, high=bias_bound))

    negative_slope = math.sqrt(5)
    gain = math.sqrt(2.0 / (1 + negative_slope**2))
    std = gain / math.sqrt(fan_in)
    weight_bound = math.sqrt(3.0) * std
    fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(
	low=-weight_bound, high=weight_bound))

    if not with_bias:
        fc_bias_attr = False

    return nn.Linear(
        input_size, hidden_size, weight_attr=fc_w_attr, bias_attr=fc_bias_attr)

def norm_layer(norm_type, nc):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = batch_norm_1d(nc)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act_type.lower()
    if act == 'relu':
        layer = nn.ReLU()
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'swish':
        layer = nn.Swish()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class MLP(paddle.nn.Sequential):
    def __init__(self, channels, act='swish', norm=None, bias=True, drop=0., last_lin=False):
        m = []

        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[i]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)