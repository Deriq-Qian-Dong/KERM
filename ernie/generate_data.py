import sys

sys.path.append("..")
from collections import Counter
import io
import os
import six
import argparse
import logging
import paddle

paddle.device.set_device("gpu")
import paddle.fluid as F
import utils

import tokenization
import dataset_factory
from model import ErnieWithGNN, ErnieRanker, ErnieWithGNNv2
import paddle.distributed as dist
from multiprocessing import cpu_count
import numpy as np
import pickle as pkl
from msmarco_eval import get_mrr
from paddle.io import DistributedBatchSampler, BatchSampler
from msmarco_eval import Mrr
from paddle.hapi.model import _all_gather
from paddle.fluid.dygraph.parallel import ParallelEnv
from tqdm import tqdm
import time


# if six.PY3:
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def define_args():
    parser = argparse.ArgumentParser('kg-ERNIE-rerank model')
    parser.add_argument('--run', type=str, default="nce")
    parser.add_argument('--model', type=str, default="ErnieWithGNN")
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    parser.add_argument('--ernie_config_file', type=str, default="/home/user/codes/kgreranker/base/ernie_config.json")
    parser.add_argument('--vocab_file', type=str, default="/home/user/codes/kgreranker/base/vocab.txt")
    parser.add_argument('--train_input_file', type=str, default="/home/user/hdfs_data/data_train/train.top2000.gz")
    parser.add_argument('--dev_input_file', type=str, default="/home/user/hdfs_data/data_train/dev.top2000.gz")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dev_batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=160)
    parser.add_argument('--warm_start_from', type=str, default="/home/user/hdfs_data/data_train/ernie_base.p")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--eval_step_proportion', type=float, default=0.33)
    parser.add_argument('--report', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--qrels', type=str, default="/home/user/hdfs_data/data_train/qrels.train.tsv")
    parser.add_argument('--top1000', type=str, default="/home/user/hdfs_data/data_train/train.qidpid.gz")
    parser.add_argument('--collection', type=str, default="/home/user/hdfs_data/data_train/collection.tsv")
    parser.add_argument('--query', type=str, default="/home/user/hdfs_data/data_train/train.query.txt")
    parser.add_argument('--min_index', type=int, default=25)
    parser.add_argument('--max_index', type=int, default=768)
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--partial_no', type=int, default=1)
    parser.add_argument('--generate_type', type=str, default="pretrain")

    # gnn config
    parser.add_argument('--with_efeat', type=bool, default=True)
    parser.add_argument('--virtual_node', type=bool, default=False)
    parser.add_argument('--num_conv_layers', type=int, default=3)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    parser.add_argument('--norm', type=str, default='layer')
    parser.add_argument('--aggr', type=str, default='softmax')
    parser.add_argument('--learn_t', type=bool, default=False)
    parser.add_argument('--learn_p', type=bool, default=False)
    parser.add_argument('--init_t', type=float, default=1.0)
    parser.add_argument('--init_p', type=float, default=1.0)
    parser.add_argument('--concat', type=bool, default=True)
    parser.add_argument('--mlp_layers', type=int, default=1)
    parser.add_argument('--edge_num', type=int, default=25)

    parser.add_argument('--resource', type=str, default="/home/user/hdfs_data/concept_net/concept.txt")
    parser.add_argument('--cpnet', type=str, default="/home/user/hdfs_data/concept_net/conceptnet.en.pruned.graph")
    parser.add_argument('--pattern_path', type=str, default="/home/user/hdfs_data/concept_net/matcher_patterns.json")
    parser.add_argument('--word2vec', type=str,
                        default="/home/user/hdfs_data/data_train/GoogleNews-vectors-negative300.bin.gz")
    parser.add_argument('--topk_sents', type=int, default=1)
    parser.add_argument('--ent_emb', type=str, default="/home/user/hdfs_data/concept_net/glove.transe.sgd.ent.npy")
    parser.add_argument('--rel_emb', type=str, default="/home/user/hdfs_data/concept_net/glove.transe.sgd.rel.npy")
    parser.add_argument('--gnn_hidden_size', type=int, default=100)
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args


def generate_eval():
    args = define_args()
    all_configs = utils.parse_file(args.ernie_config_file)
    all_configs.update(vars(args))
    all_configs = utils.HParams(**all_configs)
    cfg = all_configs
    all_configs.print_config()
    local_rank = paddle.distributed.get_rank()
    _nranks = ParallelEnv().nranks
    dataset = dataset_factory.GenEvalErnieConceptDataset(cfg)
    sampler = DistributedBatchSampler(dataset, batch_size=cfg['dev_batch_size'], shuffle=False)
    loader = paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset._collate_fn_gen, num_workers=16)
    dist.init_parallel_env()
    os.makedirs('gen_data/', exist_ok=True)
    all_steps = len(loader)
    step = 0
    start = time.time()
    local_start = time.time()
    for batch in loader:
        step += 1
        batch = [b.numpy() if i < 5 else b for i, b in enumerate(batch)]
        pkl.dump(batch, open("gen_data/dev_sample_%d_%d.pkl" % (local_rank, step), "wb"))
        if step % args.report == 0 and local_rank == 0:
            seconds = time.time() - local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print("step: %d/%d, " % (step, all_steps), "report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
            seconds = time.time() - start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))


def generate_train():
    args = define_args()
    all_configs = utils.parse_file(args.ernie_config_file)
    all_configs.update(vars(args))
    all_configs = utils.HParams(**all_configs)
    cfg = all_configs
    all_configs.print_config()
    local_rank = paddle.distributed.get_rank()
    _nranks = ParallelEnv().nranks
    # dataset=dataset_factory.GenTrainV2ErnieConceptDataset(cfg)
    dataset = dataset_factory.GenTrainErnieConceptDataset(cfg)
    sampler = DistributedBatchSampler(dataset, batch_size=1, shuffle=False)
    loader = paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset._collate_fn_gen, num_workers=10)
    dist.init_parallel_env()
    os.makedirs('gen_data/', exist_ok=True)
    all_steps = len(loader)
    step = 0
    start = time.time()
    local_start = time.time()
    for batch in loader:
        step += 1
        batch = [b.numpy() if i < 4 else b for i, b in enumerate(batch)]
        pkl.dump(batch, open("gen_data/train_sample_%d_%d.pkl" % (local_rank, step), "wb"))
        if step % args.report == 0 and local_rank == 0:
            seconds = time.time() - local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print("step: %d/%d, " % (step, all_steps), "report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
            seconds = time.time() - start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))


def generate_pretrain():
    args = define_args()
    all_configs = utils.parse_file(args.ernie_config_file)
    all_configs.update(vars(args))
    all_configs = utils.HParams(**all_configs)
    cfg = all_configs
    all_configs.print_config()
    local_rank = paddle.distributed.get_rank()
    _nranks = ParallelEnv().nranks
    partial_passage = 8841823 // 8 + 1
    partial_no = all_configs['partial_no']
    dataset = dataset_factory.GenPretrainedConceptDataset(cfg, start=partial_no * partial_passage,
                                                          end=partial_no * partial_passage + partial_passage)
    sampler = DistributedBatchSampler(dataset, batch_size=1, shuffle=False)
    loader = paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset._collate_fn_gen,
                                  num_workers=cpu_count() // _nranks)
    dist.init_parallel_env()
    os.makedirs('gen_data/', exist_ok=True)
    all_steps = len(loader)
    save_id = 0
    start = time.time()
    local_start = time.time()
    for step, batch in enumerate(loader):
        batch = batch[0]
        tmp = list(zip(*batch))
        for sample in tmp:
            save_id += 1
            sample = [s.numpy() if 0 < i < 3 else s for i, s in enumerate(sample)]
            pkl.dump(sample, open("gen_data/pretrain_sample_%d_%d.pkl" % (local_rank, save_id), "wb"))
        if step % args.report == 0 and local_rank == 0:
            seconds = time.time() - local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print("step: %d/%d, " % (step, all_steps), "report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
            seconds = time.time() - start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))


if __name__ == "__main__":
    args = define_args()
    all_configs = utils.parse_file(args.ernie_config_file)
    all_configs.update(vars(args))
    all_configs = utils.HParams(**all_configs)
    cfg = all_configs
    if cfg['generate_type'] == 'pretrain':
        generate_pretrain()
    if cfg['generate_type'] == 'eval':
        generate_eval()
    if cfg['generate_type'] == 'train':
        generate_train()
