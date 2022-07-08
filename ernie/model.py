import numpy as np

import paddle
import paddle as P
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
import pgl.nn as gnn
from pgl.utils.logger import log

import gnn_layer as L 
from modeling_ernie import *
from modeling_ernie import ErnieModel,PretrainedModel,append_name
from modeling_ernie import _build_ln,_build_linear, _get_rel_pos_bias
class LiteGEM(nn.Layer):
    def __init__(self, config):
        super(LiteGEM, self).__init__()
        log.info("gnn_type is %s" % self.__class__.__name__)

        self.config = config
        self.with_efeat = config['with_efeat']
        self.num_layers = config['num_conv_layers']
        self.drop_ratio = config['drop_ratio']
        self.virtual_node = config['virtual_node']
        self.emb_dim = config['gnn_hidden_size']
        self.norm = config['norm']

        self.gnns = paddle.nn.LayerList()
        self.norms = paddle.nn.LayerList()

        for layer in range(self.num_layers):
            self.gnns.append(L.LiteGEMConv(config, with_efeat=self.with_efeat))
            self.norms.append(L.norm_layer(self.norm, self.emb_dim))


        self.pool = gnn.GraphPool(pool_type="sum")


    def forward(self, g):
        h = g.node_feat["feature"]
        edge_emb = g.edge_feat["edge_feature"]
        h = self.gnns[0](g, h, edge_emb)

        #  print("h0: ", np.sum(h.numpy()))
        for layer in range(1, self.num_layers):
            h1 = self.norms[layer-1](h)
            h2 = F.swish(h1)
            h2 = F.dropout(h2, p=self.drop_ratio, training=self.training)
            h = self.gnns[layer](g, h2, edge_emb) + h
        h = self.norms[self.num_layers-1](h)
        h = F.dropout(h, p=self.drop_ratio, training=self.training)
        h = self.pool(g, h)
        g.node_feat["feature"] = h 
        return h,g



class ErnieWithGNNBlock(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithGNNBlock, self).__init__()
        self.cfg = cfg
        d_model = cfg['hidden_size']
        self.attn = AttentionLayer(
            cfg, name=append_name(name, 'multi_head_att'))
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'))
        self.ffn = PositionwiseFeedForwardLayer(cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'))
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)
        self.gnn = LiteGEM(cfg)
        self.batch_size = cfg['batch_size']
        self.num_word = cfg['max_seq_len']

    def forward(self, batched_graph, batch_size, attn_bias=None, past_cache=None):
        '''
        inputs:[batch_size*num_word+knowlege_nodes, 768]
        '''
        inputs = self.gnn(batched_graph)  # [-1, 768]
        inputs = inputs[:batch_size*self.num_word].reshape((batch_size, self.num_word, self.cfg['hidden_size']))
        attn_out, cache = self.attn(
            inputs, inputs, inputs, attn_bias,
            past_cache=past_cache)  #self attn 
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm
        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        batched_graph.node_feat['feature'][:batch_size*self.num_word] = hidden.reshape((-1, self.cfg['hidden_size']))
        return batched_graph, cache

class ErnieWithoutGNNBlock(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithoutGNNBlock, self).__init__()
        self.cfg = cfg
        d_model = cfg['hidden_size']
        self.attn = AttentionLayer(
            cfg, name=append_name(name, 'multi_head_att'))
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'))
        self.ffn = PositionwiseFeedForwardLayer(cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'))
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)
        self.batch_size = cfg['batch_size']
        self.num_word = cfg['max_seq_len']

    def forward(self, batched_graph, batch_size, attn_bias=None, past_cache=None):
        '''
        inputs:[batch_size*num_word+knowlege_nodes, 768]
        '''
        inputs = batched_graph.node_feat["feature"][:batch_size*self.num_word].reshape((batch_size, self.num_word, self.cfg['hidden_size']))
        attn_out, cache = self.attn(
            inputs, inputs, inputs, attn_bias,
            past_cache=past_cache)  #self attn 
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm
        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        batched_graph.node_feat['feature'][:batch_size*self.num_word] = hidden.reshape((-1, self.cfg['hidden_size']))
        return batched_graph, cache

class ErnieWithGNNEncoderStack(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithGNNEncoderStack, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = nn.LayerList(
            [ErnieWithoutGNNBlock(cfg, append_name(name, 'layer_%d' % i)) for i in range(n_layers-cfg['num_gnn_layers'])]+
            [ErnieWithGNNBlock(cfg, append_name(name, 'layer_%d' % i)) for i in range(n_layers-cfg['num_gnn_layers'], n_layers)]
            )
        self.batch_size = cfg['batch_size']
        self.num_word = cfg['max_seq_len']
        self.cfg = cfg
        
    def forward(self, batched_graph, batch_size, attn_bias=None, past_cache=None):
        if past_cache is not None:
            assert isinstance(
                past_cache, tuple
            ), 'unknown type of `past_cache`, expect tuple or list. got %s' % repr(
                type(past_cache))
            past_cache = list(zip(*past_cache))
        else:
            past_cache = [None] * len(self.block)
        cache_list_k, cache_list_v, hidden_list = [], [], [batched_graph]

        for b, p in zip(self.block, past_cache):
            batched_graph, cache = b(batched_graph, batch_size, attn_bias=attn_bias, past_cache=p)
            cache_k, cache_v = cache
            cache_list_k.append(cache_k)
            cache_list_v.append(cache_v)
            hidden_list.append(batched_graph)

        return batched_graph, hidden_list, (cache_list_k, cache_list_v)



class ErnieWithGNN(nn.Layer, PretrainedModel):
    def __init__(self, cfg, name=None):
        """
        Fundamental pretrained Ernie model
        """
        log.debug('init ErnieModel with config: %s' % repr(cfg))
        nn.Layer.__init__(self)
        self.cfg = cfg
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        d_sent = cfg.get("sent_type_vocab_size") or cfg['type_vocab_size']
        self.d_rel_pos = cfg.get('rel_pos_size', None)
        max_seq_len = cfg.get("max_seq_len", 512)
        self.n_head = cfg['num_attention_heads']
        self.return_additional_info = cfg.get('return_additional_info', False)
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        if self.d_rel_pos:
            self.rel_pos_bias = _get_rel_pos_bias(max_seq_len) 

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.edge_emb = nn.Embedding(
            cfg['edge_num'],
            cfg['hidden_size'],            
            weight_attr=P.ParamAttr(
                name=append_name(name, 'edge_embedding'),
                initializer=initializer))
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer))
        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=initializer))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)
                                               
        self.encoder_stack = ErnieWithGNNEncoderStack(cfg, append_name(name, 'encoder'))
        self.cls2score = _build_linear(
                cfg['hidden_size'],
                cfg['num_labels'],
                append_name(name, 'pooled_fc'),
                initializer)
        self.train()

    #FIXME:remove this
    def eval(self):
        if P.in_dynamic_mode():
            super(ErnieWithGNN, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        if P.in_dynamic_mode():
            super(ErnieWithGNN, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self

    def forward(self, src_ids, sent_ids, batched_graph_construction,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
        batch_size = src_ids.shape[0]
        src_ids = paddle.to_tensor(src_ids)
        sent_ids = paddle.to_tensor(sent_ids)
        assert len(
            src_ids.
            shape) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (
                repr(src_ids.shape))
        assert attn_bias is not None if past_cache else True, 'if `past_cache` is specified; attn_bias should not be None'
        d_seqlen = P.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = P.arange(
                0, d_seqlen, 1, dtype='int32').reshape([1, -1]).cast('int64')
        if attn_bias is None:
            if input_mask is None:
                input_mask = P.cast(src_ids != 0, 'float32')
            assert len(input_mask.shape) == 2
            input_mask = input_mask.unsqueeze(-1)
            attn_bias = input_mask.matmul(input_mask, transpose_y=True)
            if use_causal_mask:
                sequence = P.reshape(
                    P.arange(
                        0, d_seqlen, 1, dtype='float32') + 1., [1, 1, -1, 1])
                causal_mask = (sequence.matmul(
                    1. / sequence, transpose_y=True) >= 1.).cast('float32')
                attn_bias *= causal_mask
        else:
            assert len(
                attn_bias.shape
            ) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape
        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1).tile(
            [1, self.n_head, 1, 1])  # avoid broadcast =_=
        attn_bias.stop_gradient=True
        if self.d_rel_pos:
            rel_pos_ids = self.rel_pos_bias[:d_seqlen, :d_seqlen]
            rel_pos_ids = P.to_tensor(rel_pos_ids, dtype='int64')
            rel_pos_bias = self.rel_pos_bias_emb(rel_pos_ids).transpose([2, 0, 1])
            attn_bias += rel_pos_bias
        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        sent_embedded = self.sent_emb(sent_ids)
        embedded = src_embedded + pos_embedded + sent_embedded


        embedded = self.dropout(self.ln(embedded))

        edges, edges_type, nodes_tokens, num_nodes = batched_graph_construction
        nodes_tokens.append(0)
        num_nodes+=1
        feature = paddle.concat([embedded.reshape((-1, self.cfg['hidden_size'])), self.word_emb(paddle.to_tensor(nodes_tokens))], axis=0)
        edge_feature = self.edge_emb(paddle.to_tensor(edges_type))

        batched_graph = pgl.Graph(num_nodes=num_nodes,
            edges=edges,
            node_feat={"feature": feature},
            edge_feat={"edge_feature":edge_feature})
        out, _, _ = self.encoder_stack(batched_graph, batch_size)
        out = out.node_feat['feature'][:batch_size*self.cfg['max_seq_len']].reshape((batch_size, self.cfg['max_seq_len'], self.cfg['hidden_size']))
        cls_feature = out[:,0,:]
        return self.cls2score(cls_feature)


class ErnieWithGNNv2(nn.Layer, PretrainedModel):
    def __init__(self, cfg, name=None):
        """
        Fundamental pretrained Ernie model
        """
        log.debug('init ErnieModel with config: %s' % repr(cfg))
        nn.Layer.__init__(self)
        self.cfg = cfg
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        d_sent = cfg.get("sent_type_vocab_size") or cfg['type_vocab_size']
        self.d_rel_pos = cfg.get('rel_pos_size', None)
        max_seq_len = cfg.get("max_seq_len", 512)
        self.n_head = cfg['num_attention_heads']
        self.return_additional_info = cfg.get('return_additional_info', False)
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        if self.d_rel_pos:
            self.rel_pos_bias = _get_rel_pos_bias(max_seq_len) 

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.edge_emb = nn.Embedding(
            cfg['edge_num'],
            cfg['hidden_size'],            
            weight_attr=P.ParamAttr(
                name=append_name(name, 'edge_embedding'),
                initializer=initializer))
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer))
        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=initializer))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)
        cfg['num_gnn_layers'] = 0
        self.encoder_stack = ErnieWithGNNEncoderStack(cfg, append_name(name, 'encoder'))
        self.knowlege_encoder = LiteGEM(cfg)
        self.score = _build_linear(
                cfg['hidden_size']*2,
                cfg['num_labels'],
                append_name(name, 'pooled_fc'),
                initializer)
        self.train()

    #FIXME:remove this
    def eval(self):
        if P.in_dynamic_mode():
            super(ErnieWithGNNv2, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        if P.in_dynamic_mode():
            super(ErnieWithGNNv2, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self

    def forward(self, src_ids, sent_ids, batched_graph_construction,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
        batch_size = src_ids.shape[0]
        src_ids = paddle.to_tensor(src_ids)
        sent_ids = paddle.to_tensor(sent_ids)
        assert len(
            src_ids.
            shape) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (
                repr(src_ids.shape))
        assert attn_bias is not None if past_cache else True, 'if `past_cache` is specified; attn_bias should not be None'
        d_seqlen = P.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = P.arange(
                0, d_seqlen, 1, dtype='int32').reshape([1, -1]).cast('int64')
        if attn_bias is None:
            if input_mask is None:
                input_mask = P.cast(src_ids != 0, 'float32')
            assert len(input_mask.shape) == 2
            input_mask = input_mask.unsqueeze(-1)
            attn_bias = input_mask.matmul(input_mask, transpose_y=True)
            if use_causal_mask:
                sequence = P.reshape(
                    P.arange(
                        0, d_seqlen, 1, dtype='float32') + 1., [1, 1, -1, 1])
                causal_mask = (sequence.matmul(
                    1. / sequence, transpose_y=True) >= 1.).cast('float32')
                attn_bias *= causal_mask
        else:
            assert len(
                attn_bias.shape
            ) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape
        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1).tile(
            [1, self.n_head, 1, 1])  # avoid broadcast =_=
        attn_bias.stop_gradient=True
        if self.d_rel_pos:
            rel_pos_ids = self.rel_pos_bias[:d_seqlen, :d_seqlen]
            rel_pos_ids = P.to_tensor(rel_pos_ids, dtype='int64')
            rel_pos_bias = self.rel_pos_bias_emb(rel_pos_ids).transpose([2, 0, 1])
            attn_bias += rel_pos_bias
        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        sent_embedded = self.sent_emb(sent_ids)
        embedded = src_embedded + pos_embedded + sent_embedded


        embedded = self.dropout(self.ln(embedded))

        edges, edges_type, nodes_tokens, num_nodes = batched_graph_construction
        nodes_tokens.append(0)
        num_nodes+=1
        feature = paddle.concat([embedded.reshape((-1, self.cfg['hidden_size'])), self.word_emb(paddle.to_tensor(nodes_tokens))], axis=0)
        edge_feature = self.edge_emb(paddle.to_tensor(edges_type))

        batched_graph = pgl.Graph(num_nodes=num_nodes,
            edges=edges,
            node_feat={"feature": feature},
            edge_feat={"edge_feature":edge_feature})
        batched_graph, _, _ = self.encoder_stack(batched_graph, batch_size)
        out = self.knowlege_encoder(batched_graph)
        nodes = out[:batch_size*self.cfg['max_seq_len']].reshape((batch_size, self.cfg['max_seq_len'], self.cfg['hidden_size']))
        cls_feature = nodes[:,0,:]
        knowlege_feature = paddle.mean(nodes,axis=1,keepdim=False)
        total_feature = paddle.concat([cls_feature,knowlege_feature], axis=1)
        return self.score(total_feature)



class ErnieRanker(nn.Layer,PretrainedModel):
    def __init__(self, cfg, name=None):
        """
        Fundamental pretrained Ernie model
        """
        log.debug('init ErnieModel with config: %s' % repr(cfg))
        nn.Layer.__init__(self)
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        d_sent = cfg.get("sent_type_vocab_size") or cfg['type_vocab_size']
        self.d_rel_pos = cfg.get('rel_pos_size', None)
        max_seq_len = cfg.get("max_seq_len", 512)
        self.n_head = cfg['num_attention_heads']
        self.return_additional_info = cfg.get('return_additional_info', False)
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        if self.d_rel_pos:
            self.rel_pos_bias = _get_rel_pos_bias(max_seq_len) 

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer))
        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=initializer))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)

        self.encoder_stack = ErnieEncoderStack(cfg,
                                               append_name(name, 'encoder'))
        if cfg.get('has_pooler', True):
            self.pooler = _build_linear(
                cfg['hidden_size'],
                cfg['hidden_size'],
                append_name(name, 'pooled_fc'),
                initializer, )
        else:
            self.pooler = None
        self.cls2score = _build_linear(
                cfg['hidden_size'],
                cfg['num_labels'],
                append_name(name, 'pooled_fc'),
                initializer)
        self.train()

    #FIXME:remove this
    def eval(self):
        if P.in_dynamic_mode():
            super(ErnieRanker, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        if P.in_dynamic_mode():
            super(ErnieRanker, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self

    def forward(self,
                src_ids,
                sent_ids=None,
                graph=None,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
                
        """
        Args:
            src_ids (`Variable` of shape `[batch_size, seq_len]`):
                Indices of input sequence tokens in the vocabulary.
            sent_ids (optional, `Variable` of shape `[batch_size, seq_len]`):
                aka token_type_ids, Segment token indices to indicate first and second portions of the inputs.
                if None, assume all tokens come from `segment_a`
            pos_ids(optional, `Variable` of shape `[batch_size, seq_len]`):
                Indices of positions of each input sequence tokens in the position embeddings.
            input_mask(optional `Variable` of shape `[batch_size, seq_len]`):
                Mask to avoid performing attention on the padding token indices of the encoder input.
            attn_bias(optional, `Variable` of shape `[batch_size, seq_len, seq_len] or False`):
                3D version of `input_mask`, if set, overrides `input_mask`; if set not False, will not apply attention mask
            past_cache(optional, tuple of two lists: cached key and cached value,
                each is a list of `Variable`s of shape `[batch_size, seq_len, hidden_size]`):
                cached key/value tensor that will be concated to generated key/value when performing self attention.
                if set, `attn_bias` should not be None.
        Returns:
            pooled (`Variable` of shape `[batch_size, hidden_size]`):
                output logits of pooler classifier
            encoded(`Variable` of shape `[batch_size, seq_len, hidden_size]`):
                output logits of transformer stack
            info (Dictionary):
                addtional middle level info, inclues: all hidden stats, k/v caches.
        """
        assert len(
            src_ids.
            shape) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (
                repr(src_ids.shape))
        assert attn_bias is not None if past_cache else True, 'if `past_cache` is specified; attn_bias should not be None'
        d_seqlen = P.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = P.arange(
                0, d_seqlen, 1, dtype='int32').reshape([1, -1]).cast('int64')
        if attn_bias is None:
            if input_mask is None:
                input_mask = P.cast(src_ids != 0, 'float32')
            assert len(input_mask.shape) == 2
            input_mask = input_mask.unsqueeze(-1)
            attn_bias = input_mask.matmul(input_mask, transpose_y=True)
            if use_causal_mask:
                sequence = P.reshape(
                    P.arange(
                        0, d_seqlen, 1, dtype='float32') + 1., [1, 1, -1, 1])
                causal_mask = (sequence.matmul(
                    1. / sequence, transpose_y=True) >= 1.).cast('float32')
                attn_bias *= causal_mask
        else:
            assert len(
                attn_bias.shape
            ) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape
        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1).tile(
            [1, self.n_head, 1, 1])  # avoid broadcast =_=
        attn_bias.stop_gradient=True
        if sent_ids is None:
            sent_ids = P.zeros_like(src_ids)
        if self.d_rel_pos:
            rel_pos_ids = self.rel_pos_bias[:d_seqlen, :d_seqlen]
            rel_pos_ids = P.to_tensor(rel_pos_ids, dtype='int64')
            rel_pos_bias = self.rel_pos_bias_emb(rel_pos_ids).transpose([2, 0, 1])
            attn_bias += rel_pos_bias
        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        sent_embedded = self.sent_emb(sent_ids)
        embedded = src_embedded + pos_embedded + sent_embedded


        embedded = self.dropout(self.ln(embedded))

        encoded, hidden_list, cache_list = self.encoder_stack(
            embedded, attn_bias, past_cache=past_cache)
        if self.pooler is not None:
            pooled = F.tanh(self.pooler(encoded[:, 0, :]))
        else:
            pooled = None

        additional_info = {
            'hiddens': hidden_list,
            'caches': cache_list,
        }
        score = self.cls2score(pooled)
        return score



class ErnieWithConcept(nn.Layer,PretrainedModel):
    def __init__(self, cfg, name=None):
        """
        Fundamental pretrained Ernie model
        """
        log.debug('init ErnieModel with config: %s' % repr(cfg))
        nn.Layer.__init__(self)
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        d_sent = cfg.get("sent_type_vocab_size") or cfg['type_vocab_size']
        self.d_rel_pos = cfg.get('rel_pos_size', None)
        max_seq_len = cfg.get("max_seq_len", 512)
        self.n_head = cfg['num_attention_heads']
        self.return_additional_info = cfg.get('return_additional_info', False)
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        if self.d_rel_pos:
            self.rel_pos_bias = _get_rel_pos_bias(max_seq_len) 

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer))
        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=initializer))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)

        self.encoder_stack = ErnieEncoderStack(cfg,
                                               append_name(name, 'encoder'))
        if cfg.get('has_pooler', True):
            self.pooler = _build_linear(
                cfg['hidden_size'],
                cfg['gnn_hidden_size'],
                append_name(name, 'pooled_fc'),
                initializer, )
        else:
            self.pooler = None
        self.gnn = LiteGEM(cfg)
        self.knowlege_pooler = _build_linear(
                cfg['gnn_hidden_size'],
                cfg['gnn_hidden_size'],
                append_name(name, 'knowledge_pooled_fc'),
                initializer)
        self.cls2score = _build_linear(
                cfg['gnn_hidden_size']*2,
                cfg['num_labels'],
                append_name(name, 'pooled_fc'),
                initializer)
        self.train()

    #FIXME:remove this
    def eval(self):
        if P.in_dynamic_mode():
            super(ErnieWithConcept, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        if P.in_dynamic_mode():
            super(ErnieWithConcept, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self

    def forward(self,
                src_ids,
                sent_ids=None,
                graph=None,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
                
        """
        Args:
            src_ids (`Variable` of shape `[batch_size, seq_len]`):
                Indices of input sequence tokens in the vocabulary.
            sent_ids (optional, `Variable` of shape `[batch_size, seq_len]`):
                aka token_type_ids, Segment token indices to indicate first and second portions of the inputs.
                if None, assume all tokens come from `segment_a`
            pos_ids(optional, `Variable` of shape `[batch_size, seq_len]`):
                Indices of positions of each input sequence tokens in the position embeddings.
            input_mask(optional `Variable` of shape `[batch_size, seq_len]`):
                Mask to avoid performing attention on the padding token indices of the encoder input.
            attn_bias(optional, `Variable` of shape `[batch_size, seq_len, seq_len] or False`):
                3D version of `input_mask`, if set, overrides `input_mask`; if set not False, will not apply attention mask
            past_cache(optional, tuple of two lists: cached key and cached value,
                each is a list of `Variable`s of shape `[batch_size, seq_len, hidden_size]`):
                cached key/value tensor that will be concated to generated key/value when performing self attention.
                if set, `attn_bias` should not be None.
        Returns:
            pooled (`Variable` of shape `[batch_size, hidden_size]`):
                output logits of pooler classifier
            encoded(`Variable` of shape `[batch_size, seq_len, hidden_size]`):
                output logits of transformer stack
            info (Dictionary):
                addtional middle level info, inclues: all hidden stats, k/v caches.
        """
        assert len(
            src_ids.
            shape) == 2, 'expect src_ids.shape = [batch, sequecen], got %s' % (
                repr(src_ids.shape))
        assert attn_bias is not None if past_cache else True, 'if `past_cache` is specified; attn_bias should not be None'
        d_seqlen = P.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = P.arange(
                0, d_seqlen, 1, dtype='int32').reshape([1, -1]).cast('int64')
        if attn_bias is None:
            if input_mask is None:
                input_mask = P.cast(src_ids != 0, 'float32')
            assert len(input_mask.shape) == 2
            input_mask = input_mask.unsqueeze(-1)
            attn_bias = input_mask.matmul(input_mask, transpose_y=True)
            if use_causal_mask:
                sequence = P.reshape(
                    P.arange(
                        0, d_seqlen, 1, dtype='float32') + 1., [1, 1, -1, 1])
                causal_mask = (sequence.matmul(
                    1. / sequence, transpose_y=True) >= 1.).cast('float32')
                attn_bias *= causal_mask
        else:
            assert len(
                attn_bias.shape
            ) == 3, 'expect attn_bias tobe rank 3, got %r' % attn_bias.shape
        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1).tile(
            [1, self.n_head, 1, 1])  # avoid broadcast =_=
        attn_bias.stop_gradient=True
        if sent_ids is None:
            sent_ids = P.zeros_like(src_ids)
        if self.d_rel_pos:
            rel_pos_ids = self.rel_pos_bias[:d_seqlen, :d_seqlen]
            rel_pos_ids = P.to_tensor(rel_pos_ids, dtype='int64')
            rel_pos_bias = self.rel_pos_bias_emb(rel_pos_ids).transpose([2, 0, 1])
            attn_bias += rel_pos_bias
        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        sent_embedded = self.sent_emb(sent_ids)
        embedded = src_embedded + pos_embedded + sent_embedded


        embedded = self.dropout(self.ln(embedded))

        encoded, hidden_list, cache_list = self.encoder_stack(
            embedded, attn_bias, past_cache=past_cache)
        if self.pooler is not None:
            pooled = F.tanh(self.pooler(encoded[:, 0, :]))
        else:
            pooled = None
        knowledge,graph = self.gnn(graph)
        knowledge_pooled = F.tanh(self.knowlege_pooler(knowledge))
        score = self.cls2score(paddle.concat([pooled,knowledge_pooled],axis=1))
        return score

class ConceptOnly(nn.Layer):
    def __init__(self, cfg, name=None):
        nn.Layer.__init__(self)
        self.gnn = LiteGEM(cfg)
        self.cfg = cfg
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        self.score = _build_linear(
                cfg['gnn_hidden_size'],
                cfg['num_labels'],
                append_name(name, 'score'),
                initializer)
        self.train()
    #FIXME:remove this
    def eval(self):
        if P.in_dynamic_mode():
            super(ConceptOnly, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self

    def train(self):
        if P.in_dynamic_mode():
            super(ConceptOnly, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self
    
    def forward(self, graph):
        out, graph = self.gnn(graph)
        score = self.score(out)
        return score
