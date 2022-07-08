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
from modeling_ernie import ErnieModel,PretrainedModel,append_name,ACT_DICT
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
        g.node_feat["feature"] = h+g.node_feat["feature"]
        return g

class PositionwiseFeedForwardLayer(nn.Layer):
    def __init__(self, cfg, name=None):
        super(PositionwiseFeedForwardLayer, self).__init__()
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)
        self.act = ACT_DICT[cfg['hidden_act']]()
        self.i = _build_linear(
            d_model,
            d_ffn,
            append_name(name, 'fc_0'),
            initializer, cfg['ernie_lr'])
        self.o = _build_linear(d_ffn, d_model,
                               append_name(name, 'fc_1'), initializer, cfg['ernie_lr'])
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = nn.Dropout(p=prob)
    def forward(self, inputs):
        hidden = self.act(self.i(inputs))
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out
class PositionwiseFeedForwardWithGNNLayer(nn.Layer):
    def __init__(self, cfg, name=None):
        super(PositionwiseFeedForwardWithGNNLayer, self).__init__()
        self.cfg = cfg
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)
        self.d_ffn = d_ffn
        self.act = ACT_DICT[cfg['hidden_act']]()
        self.i = _build_linear(
            d_model,
            d_ffn,
            append_name(name, 'fc_0'),
            initializer, 1.0)
        self.i_ent = _build_linear(
            cfg['gnn_hidden_size'],
            d_ffn,
            append_name(name, 'fc_ent_0'),
            initializer, 1.0)
        self.o = _build_linear(d_ffn, d_model,
                               append_name(name, 'fc_1'), initializer, 1.0)
        self.o_ent = _build_linear(d_ffn, cfg['gnn_hidden_size'],
                               append_name(name, 'fc_ent_1'), initializer, 1.0)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = nn.Dropout(p=prob)
        self.gnn = LiteGEM(cfg)
    def forward(self, inputs, batched_graph):
        node_feat = batched_graph.node_feat['feature']
        mask = paddle.cast(node_feat != 0, 'float64')
        text_hidden = self.i(inputs)  
        batch_size, seq_len, hidden_size = inputs.shape
        text_hidden = text_hidden.reshape((-1, self.d_ffn))  #  -1, 768*4
        text_hidden = paddle.concat([text_hidden, paddle.zeros((node_feat.shape[0]-text_hidden.shape[0], self.d_ffn))])
        ent_hidden = self.i_ent(node_feat)
        hidden = self.act(ent_hidden+text_hidden)  #  -1, 768*4
        ent_out = self.o_ent(hidden)*mask
        hidden = self.dropout(hidden)
        text_out = self.o(hidden[:batch_size*seq_len])
        text_out = text_out.reshape((batch_size, seq_len, hidden_size))
        batched_graph.node_feat['feature'] = ent_out
        batched_graph = self.gnn(batched_graph)
        # batched_graph.node_feat['feature'] = batched_graph.node_feat['feature']*mask
        return text_out, batched_graph
class ErnieWithGNNBlock(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithGNNBlock, self).__init__()
        d_model = cfg['hidden_size']

        self.attn = AttentionLayer(
            cfg, name=append_name(name, 'multi_head_att'), lr=1.0)
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'), lr=1.0)
        self.ffn = PositionwiseFeedForwardWithGNNLayer(
            cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'), lr=1.0)
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)
    def forward(self, inputs, attn_bias=None, batch_graph=None):
        attn_out, cache = self.attn(
            inputs, inputs, inputs, attn_bias,
            past_cache=None)  #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm
        ffn_out, batch_graph = self.ffn(hidden, batch_graph)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden,batch_graph


class ErnieWithGNNBlockV2(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithGNNBlockV2, self).__init__()
        d_model = cfg['hidden_size']
        self.gnn_size = cfg['gnn_hidden_size']
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        self.ernie_emb_2_transe_emb = _build_linear(
                                        d_model,
                                        cfg['gnn_hidden_size'],
                                        append_name(name, 'ernie_emb_2_transe_emb'),
                                        initializer, 1.0)
        self.transe_emb_2_ernie_emb = _build_linear(
                                        cfg['gnn_hidden_size'],
                                        d_model,
                                        append_name(name, 'transe_emb_2_ernie_emb'),
                                        initializer, 1.0)
        self.attn = AttentionLayer(
            cfg, name=append_name(name, 'multi_head_att'), lr=1.0)
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'), lr=1.0)
        self.ffn = PositionwiseFeedForwardLayer(
            cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'), lr=1.0)
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)
        self.gnn = LiteGEM(cfg)

    def forward(self, inputs, attn_bias=None, batch_graph=None):
        text_hidden = self.ernie_emb_2_transe_emb(inputs)  #把ERNIE 768-->100

        node_feat = batch_graph.node_feat['feature']  # entity transE表示
        batch_size, seq_len, hidden_size = inputs.shape
        text_hidden = text_hidden.reshape((-1, self.gnn_size))  #  -1, 100
        text_hidden = paddle.concat([text_hidden, paddle.zeros((node_feat.shape[0]-text_hidden.shape[0], self.gnn_size))])

        text_with_knowledge = text_hidden+node_feat   # 融合
        batch_graph.node_feat['feature'] = text_with_knowledge
        batch_graph = self.gnn(batch_graph)  # GMN

        text_out = self.transe_emb_2_ernie_emb(text_with_knowledge[:batch_size*seq_len])  # 100-->768后再做MHA
        inputs = text_out.reshape((batch_size, seq_len, hidden_size))  # 修改维度输入到MHA
        attn_out, cache = self.attn(
            inputs, inputs, inputs, attn_bias,
            past_cache=None)  #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm
        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden,batch_graph


class ErnieWithGNNBlockV3(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithGNNBlockV3, self).__init__()
        d_model = cfg['hidden_size']
        self.d_model = d_model
        self.gnn_size = cfg['gnn_hidden_size']
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        self.fusion_2_ernie = _build_linear(
                                        d_model,
                                        d_model,
                                        append_name(name, 'ernie_emb_2_transe_emb'),
                                        initializer, 1.0)
        self.transe_emb_2_ernie_emb = _build_linear(
                                        cfg['gnn_hidden_size'],
                                        d_model,
                                        append_name(name, 'transe_emb_2_ernie_emb'),
                                        initializer, 1.0)
        self.attn = AttentionLayer(
            cfg, name=append_name(name, 'multi_head_att'), lr=1.0)
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'), lr=1.0)
        self.ffn = PositionwiseFeedForwardLayer(
            cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'), lr=1.0)
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)

    def forward(self, inputs, attn_bias=None, batch_graph=None):
        # text_hidden = self.ernie_emb_2_transe_emb(inputs)

        node_feat = batch_graph.node_feat['feature']
        batch_size, seq_len, hidden_size = inputs.shape
        text_hidden = inputs.reshape((-1, self.d_model))  #  -1, 768
        text_hidden = paddle.concat([text_hidden, paddle.zeros((node_feat.shape[0]-text_hidden.shape[0], self.d_model))])

        text_with_knowledge = text_hidden + self.transe_emb_2_ernie_emb(node_feat)

        text_out = self.fusion_2_ernie(text_with_knowledge[:batch_size*seq_len])
        inputs = text_out.reshape((batch_size, seq_len, hidden_size))

        attn_out, cache = self.attn(
            inputs, inputs, inputs, attn_bias,
            past_cache=None)  #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm
        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden,batch_graph



class ErnieWithoutGNNBlock(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithoutGNNBlock, self).__init__()
        d_model = cfg['hidden_size']
        self.attn = AttentionLayer(
            cfg, name=append_name(name, 'multi_head_att'), lr=cfg['ernie_lr'])
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'), lr=cfg['ernie_lr'])
        self.ffn = PositionwiseFeedForwardLayer(
            cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'), lr=cfg['ernie_lr'])
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = nn.Dropout(p=prob)
    def forward(self, inputs, attn_bias=None, batch_graph=None):
        attn_out, cache = self.attn(
            inputs, inputs, inputs, attn_bias,
            past_cache=None)  #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm
        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden, batch_graph
class ErnieWithGNNEncoderStack(nn.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieWithGNNEncoderStack, self).__init__()
        n_layers = cfg['num_hidden_layers']
        kerm_version = cfg['kerm_version']
        Block_MP = {'v1':ErnieWithGNNBlock,'v2':ErnieWithGNNBlockV2,'v3':ErnieWithGNNBlockV3}
        self.block = nn.LayerList(
            [ErnieWithoutGNNBlock(cfg, append_name(name, 'layer_%d' % i)) for i in range(n_layers-cfg['num_gnn_layers'])]+
            [Block_MP[kerm_version](cfg, append_name(name, 'layer_%d' % i)) for i in range(n_layers-cfg['num_gnn_layers'], n_layers)]
            )
        self.batch_size = cfg['batch_size']
        self.num_word = cfg['max_seq_len']
        self.cfg = cfg
        
    def forward(self, inputs, attn_bias=None, batch_graph=None):
        for b in self.block:
            inputs, batch_graph = b(inputs, attn_bias=attn_bias, batch_graph=batch_graph)
        return inputs
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
        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'), lr=cfg['ernie_lr'])
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=initializer, learning_rate=cfg['ernie_lr']))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)
        self.encoder_stack = ErnieWithGNNEncoderStack(cfg,
                                               append_name(name, 'encoder'))
        if cfg.get('has_pooler', True):
            self.pooler = _build_linear(
                cfg['hidden_size'],
                cfg['hidden_size'],
                append_name(name, 'pooled_fc'),
                initializer, cfg['ernie_lr'])
        else:
            self.pooler = None
        self.cls2score = _build_linear(
                cfg['hidden_size'],
                cfg['num_labels'],
                append_name(name, 'cls2score'),
                initializer, 1.0)
        self.mlm = _build_linear(
            d_model,
            d_model,
            append_name(name, 'mask_lm_trans_fc'),
            initializer, 1.0)
        self.act = ACT_DICT[cfg['hidden_act']]()
        self.mlm_ln = _build_ln(
            d_model, name=append_name(name, 'mask_lm_trans'), lr=1.0)
        self.mlm_bias = P.create_parameter(
            dtype='float32',
            shape=[d_vocab],
            attr=P.ParamAttr(
                name=append_name(name, 'mask_lm_out_fc.b_0'),
                initializer=nn.initializer.Constant(value=0.0)),
            is_bias=True, )   
        self.train()
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
                batch_graph=None,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False,
                with_mlm=False,
                mlm_pos=None):
                
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
        batch_graph = batch_graph.tensor()
        batch_graph.node_feat['feature'] = batch_graph.node_feat['feature'].astype('float32')
        encoded = self.encoder_stack(
            embedded, attn_bias, batch_graph=batch_graph)
        if self.pooler is not None:
            pooled = F.tanh(self.pooler(encoded[:, 0, :]))
        else:
            pooled = None
        score = self.cls2score(pooled)
        if with_mlm:
            encoded_2d = encoded.gather_nd(mlm_pos)
            encoded_2d = self.act(self.mlm(encoded_2d))
            encoded_2d = self.mlm_ln(encoded_2d)
            logits_2d = encoded_2d.matmul(
                self.word_emb.weight, transpose_y=True) + self.mlm_bias
            return score, logits_2d
        return score


class ErnieWithConceptv2(nn.Layer,PretrainedModel):
    def __init__(self, cfg, name=None):
        """
        Fundamental pretrained Ernie model
        """
        log.debug('init ErnieModel with config: %s' % repr(cfg))
        nn.Layer.__init__(self)
        self.cfg = cfg
        self.num_node = cfg['batch_size']*cfg['max_seq_len']*cfg['sample_num']
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
        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'), lr=cfg['ernie_lr'])
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=initializer, learning_rate=cfg['ernie_lr']))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)
        self.encoder_stack = ErnieWithGNNEncoderStack(cfg,
                                               append_name(name, 'encoder'))
        self.gnn = LiteGEM(cfg)
        if cfg.get('has_pooler', True):
            self.pooler = _build_linear(
                cfg['hidden_size'],
                cfg['hidden_size'],
                append_name(name, 'pooled_fc'),
                initializer, cfg['ernie_lr'])
        else:
            self.pooler = None
        self.knowledge_pooler = _build_linear(
                cfg['gnn_hidden_size'],
                cfg['hidden_size'],
                append_name(name, 'knowledge_pooled_fc'),
                initializer, 1.0)
        self.cls2score = _build_linear(
                cfg['hidden_size']*2,
                cfg['num_labels'],
                append_name(name, 'cls2score'),
                initializer, 1.0)
        self.train()
    #FIXME:remove this
    def eval(self):
        if P.in_dynamic_mode():
            super(ErnieWithConceptv2, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self
    def train(self):
        if P.in_dynamic_mode():
            super(ErnieWithConceptv2, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self
    def forward(self,
                src_ids,
                sent_ids=None,
                batch_graph=None,
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
        b_size = P.shape(src_ids)[0]
        num_node = b_size*d_seqlen
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
        batch_graph = batch_graph.tensor()
        batch_graph.node_feat['feature'] = batch_graph.node_feat['feature'].astype('float32')
        encoded = self.encoder_stack(
            embedded, attn_bias, batch_graph=batch_graph)
        batch_graph = self.gnn(batch_graph)
        knowledge_out = batch_graph.node_feat["feature"][:num_node].reshape((b_size,d_seqlen,self.cfg['gnn_hidden_size']))
        knowledge_out = self.knowledge_pooler(paddle.mean(knowledge_out, axis=1))
        if self.pooler is not None:
            pooled = F.tanh(self.pooler(encoded[:, 0, :]))
        else:
            pooled = None
        pooled = paddle.concat([pooled, knowledge_out], axis=1)
        score = self.cls2score(pooled)
        return score



class NSPHead(nn.Layer):
    def __init__(self, cfg, name=None):
        super(NSPHead, self).__init__()
        initializer = nn.initializer.TruncatedNormal(
            std=cfg['initializer_range'])
        self.nsp = _build_linear(cfg['hidden_size'], 3,
                                 append_name(name, 'nsp_fc'), initializer,1.0)
    def forward(self, inputs, labels):
        logits = self.nsp(inputs)
        loss = F.cross_entropy(logits, labels)
        return loss

class pretrainedErnieWithConcept(nn.Layer,PretrainedModel):
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
        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'), lr=cfg['ernie_lr'])
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'word_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        self.pos_emb = nn.Embedding(
            d_pos,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'pos_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        self.sent_emb = nn.Embedding(
            d_sent,
            d_emb,
            weight_attr=P.ParamAttr(
                name=append_name(name, 'sent_embedding'),
                initializer=initializer, learning_rate=cfg['ernie_lr']))
        if self.d_rel_pos:
            self.rel_pos_bias_emb = nn.Embedding(
                self.d_rel_pos,
                self.n_head,
                weight_attr=P.ParamAttr(
                    name=append_name(name, 'rel_pos_embedding'),
                    initializer=initializer, learning_rate=cfg['ernie_lr']))
        prob = cfg['hidden_dropout_prob']
        self.dropout = nn.Dropout(p=prob)
        self.encoder_stack = ErnieWithGNNEncoderStack(cfg,
                                               append_name(name, 'encoder'))
        if cfg.get('has_pooler', True):
            self.pooler = _build_linear(
                cfg['hidden_size'],
                cfg['hidden_size'],
                append_name(name, 'pooled_fc'),
                initializer, cfg['ernie_lr'])
        else:
            self.pooler = None
        self.cls2score = _build_linear(
                cfg['hidden_size'],
                cfg['num_labels'],
                append_name(name, 'cls2score'),
                initializer, 1.0)

        self.pooler_heads = nn.LayerList([NSPHead(cfg, name=name)])
        self.mlm = _build_linear(
            d_model,
            d_model,
            append_name(name, 'mask_lm_trans_fc'),
            initializer, 1.0)
        self.act = ACT_DICT[cfg['hidden_act']]()
        self.mlm_ln = _build_ln(
            d_model, name=append_name(name, 'mask_lm_trans'), lr=1.0)
        self.mlm_bias = P.create_parameter(
            dtype='float32',
            shape=[d_vocab],
            attr=P.ParamAttr(
                name=append_name(name, 'mask_lm_out_fc.b_0'),
                initializer=nn.initializer.Constant(value=0.0)),
            is_bias=True, )

        self.train()
    #FIXME:remove this
    def eval(self):
        if P.in_dynamic_mode():
            super(pretrainedErnieWithConcept, self).eval()
        self.training = False
        for l in self.sublayers():
            l.training = False
        return self
    def train(self):
        if P.in_dynamic_mode():
            super(pretrainedErnieWithConcept, self).train()
        self.training = True
        for l in self.sublayers():
            l.training = True
        return self


    def forward(self,
                srp_labels,
                mlm_pos,
                mlm_labels,
                src_ids,
                sent_ids=None,
                batch_graph=None,
                pos_ids=None,
                input_mask=None,
                attn_bias=None,
                past_cache=None,
                use_causal_mask=False):
        #srp_labels, pair_input_ids, pair_token_type_ids, mask_pos, mask_label, g
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
        batch_graph = batch_graph.tensor()
        batch_graph.node_feat['feature'] = batch_graph.node_feat['feature'].astype('float32')
        encoded = self.encoder_stack(
            embedded, attn_bias, batch_graph=batch_graph)
        if self.pooler is not None:
            pooled = F.tanh(self.pooler(encoded[:, 0, :]))
        else:
            pooled = None
        
        nsp_loss = self.pooler_heads[0](pooled, srp_labels)

        encoded_2d = encoded.gather_nd(mlm_pos)
        encoded_2d = self.act(self.mlm(encoded_2d))
        encoded_2d = self.mlm_ln(encoded_2d)
        logits_2d = encoded_2d.matmul(
            self.word_emb.weight, transpose_y=True) + self.mlm_bias
        mlm_loss = F.cross_entropy(logits_2d, mlm_labels)
        total_loss = mlm_loss + nsp_loss
        return total_loss

