import os
import pickle
import paddle
import numpy as np
def match_embedding_param(convert_parameter_name_dict):
    convert_parameter_name_dict[
        "word_emb.weight"] = "word_embedding"
    convert_parameter_name_dict[
        "pos_emb.weight"] = "pos_embedding"
    convert_parameter_name_dict[
        "sent_emb.weight"] = "sent_embedding"
    convert_parameter_name_dict[
        "ln.weight"] = "pre_encoder_layer_norm_scale"
    convert_parameter_name_dict[
        "ln.bias"] = "pre_encoder_layer_norm_bias"
    return convert_parameter_name_dict


def match_encoder_param(convert_parameter_name_dict, layer_num=4):
    dygraph_proj_names = ["q", "k", "v", "o"]
    static_proj_names = ["query", "key", "value", "output"]
    dygraph_param_names = ["weight", "bias"]
    static_param_names = ["w", "b"]
    dygraph_layer_norm_param_names = ["weight", "bias"]
    static_layer_norm_param_names = ["scale", "bias"]

    # Firstly, converts the multihead_attention to the parameter.
#     dygraph_format_name = "encoder.layers.{}.self_attn.{}_proj.{}"
    # encoder_stack.block.10.attn.o.bias
    dygraph_format_name = "encoder_stack.block.{}.attn.{}.{}"
    static_format_name = "encoder_layer_{}_multi_head_att_{}_fc.{}_0"
    for i in range(0, layer_num):
        for dygraph_proj_name, static_proj_name in zip(dygraph_proj_names,
                                                       static_proj_names):
            for dygraph_param_name, static_param_name in zip(
                    dygraph_param_names, static_param_names):
                convert_parameter_name_dict[dygraph_format_name.format(i, dygraph_proj_name, dygraph_param_name)] = \
                    static_format_name.format(i, static_proj_name, static_param_name)

    # Secondly, converts the encoder ffn parameter.     
#     dygraph_ffn_linear_format_name = "encoder.layers.{}.linear{}.{}"
    #encoder_stack.block.0.ffn.i.weight
    dygraph_ffn_linear_format_name = "encoder_stack.block.{}.ffn.{}.{}"
    static_ffn_linear_format_name = "encoder_layer_{}_ffn_fc_{}.{}_0"
    for i in range(0, layer_num):
        for cnt,j in enumerate(['i','o']):
            for dygraph_param_name, static_param_name in zip(
                    dygraph_param_names, static_param_names):
                convert_parameter_name_dict[dygraph_ffn_linear_format_name.format(i, j,  dygraph_param_name)] = \
                  static_ffn_linear_format_name.format(i, cnt, static_param_name)

    # Thirdly, converts the multi_head layer_norm parameter.
    # dygraph_encoder_attention_layer_norm_format_name = "encoder.layers.{}.norm1.{}"
    dygraph_encoder_attention_layer_norm_format_name = "encoder_stack.block.{}.ln1.{}"
    static_encoder_attention_layer_norm_format_name = "encoder_layer_{}_post_att_layer_norm_{}"
    for i in range(0, layer_num):
        for dygraph_param_name, static_pararm_name in zip(
                dygraph_layer_norm_param_names, static_layer_norm_param_names):
            convert_parameter_name_dict[dygraph_encoder_attention_layer_norm_format_name.format(i, dygraph_param_name)] = \
                static_encoder_attention_layer_norm_format_name.format(i, static_pararm_name)

    # dygraph_encoder_ffn_layer_norm_format_name = "encoder.layers.{}.norm2.{}"
    dygraph_encoder_ffn_layer_norm_format_name = "encoder_stack.block.{}.ln2.{}"
    static_encoder_ffn_layer_norm_format_name = "encoder_layer_{}_post_ffn_layer_norm_{}"
    for i in range(0, layer_num):
        for dygraph_param_name, static_pararm_name in zip(
                dygraph_layer_norm_param_names, static_layer_norm_param_names):
            convert_parameter_name_dict[dygraph_encoder_ffn_layer_norm_format_name.format(i, dygraph_param_name)] = \
                 static_encoder_ffn_layer_norm_format_name.format(i, static_pararm_name)
    return convert_parameter_name_dict

def match_pooler_parameter(convert_parameter_name_dict):
    convert_parameter_name_dict["pooler.dense.weight"] = "pooled_fc.w_0"
    convert_parameter_name_dict["pooler.dense.bias"] = "pooled_fc.b_0"
    return convert_parameter_name_dict


def match_mlm_parameter(convert_parameter_name_dict):
    # convert_parameter_name_dict["cls.predictions.decoder_weight"] = "word_embedding"
    convert_parameter_name_dict[
        "cls.predictions.decoder_bias"] = "mask_lm_out_fc.b_0"
    convert_parameter_name_dict[
        "cls.predictions.transform.weight"] = "mask_lm_trans_fc.w_0"
    convert_parameter_name_dict[
        "cls.predictions.transform.bias"] = "mask_lm_trans_fc.b_0"
    convert_parameter_name_dict[
        "cls.predictions.layer_norm.weight"] = "mask_lm_trans_layer_norm_scale"
    convert_parameter_name_dict[
        "cls.predictions.layer_norm.bias"] = "mask_lm_trans_layer_norm_bias"
    return convert_parameter_name_dict


def convert_static_to_dygraph_params(dygraph_params_save_path,
                                     static_params_dir,
                                     static_to_dygraph_param_name,
                                     model_name='static'):
    files = os.listdir(static_params_dir)

    state_dict = {}
    model_name = model_name
    for name in files:
        path = os.path.join(static_params_dir, name)
        # static_para_name = name.replace('@HUB_chinese-roberta-wwm-ext-large@',
        #                                 '')  # for hub module params
        static_para_name = name.replace('.npy', '')
        if static_para_name not in static_to_dygraph_param_name:
            print(static_para_name, "not in static_to_dygraph_param_name")
            continue
        dygraph_para_name = static_to_dygraph_param_name[static_para_name]
        value = np.load(path)
        if "cls" in dygraph_para_name:
            # Note: cls.predictions parameters do not need add `model_name.` prefix
            state_dict[dygraph_para_name] = value
        else:
            state_dict[model_name + '.' + dygraph_para_name] = value

    with open(dygraph_params_save_path, 'wb') as f:
        pickle.dump(state_dict, f)
    params = paddle.load(dygraph_params_save_path)

    for name in state_dict.keys():
        if name in params:
            assert ((state_dict[name] == params[name].numpy()).all() == True)
        else:
            print(name, 'not in params')


if __name__=="__main__":
    convert_parameter_name_dict = {}

    convert_parameter_name_dict = match_embedding_param(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_encoder_param(
        convert_parameter_name_dict, layer_num=24)
    convert_parameter_name_dict = match_pooler_parameter(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_mlm_parameter(
        convert_parameter_name_dict)

    static_to_dygraph_param_name = {
        value: key
        for key, value in convert_parameter_name_dict.items()
    }
    import paddle
    state_dict = paddle.load("../NAACL2021-RocketQA/checkpoint/marco_cross_encoder_large/")
    params = {}
    miss = []
    for key in state_dict:
        try:
            params[static_to_dygraph_param_name[key]] = state_dict[key]
        except:
            miss.append(key)
    