from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
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
from model import ErnieWithGNN, ErnieRanker,ErnieWithGNNv2
from ernie_concept import ErnieWithConcept,pretrainedErnieWithConcept
import paddle.distributed as dist
from multiprocessing import cpu_count
import numpy as np
import pickle as pkl
from msmarco_eval import get_mrr
from paddle.io import DistributedBatchSampler,BatchSampler,get_worker_info
from msmarco_eval import Mrr
from paddle.hapi.model import _all_gather
from paddle.fluid.dygraph.parallel import ParallelEnv
from tqdm import tqdm
import time
# if six.PY3:
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import spacy
import glob
import random
import numpy as np
import tarfile
from paths import BatchedGraphBuilder,ConceptGraphBuilder
import paddle
from paddle.io import Dataset, DistributedBatchSampler, get_worker_info,IterableDataset
import paddle.distributed as dist
import pandas as pd
import tokenization
import pickle as pkl
import glob
import pgl
import re
import multiprocessing

def define_args():
    parser = argparse.ArgumentParser('kg-ERNIE-rerank model')
    parser.add_argument('--run', type=str, default="nce")
    parser.add_argument('--model', type=str, default="ErnieWithGNN")
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    parser.add_argument('--ernie_config_file', type=str, default="/home/dongqian06/codes/kgreranker/base/ernie_config.json")
    parser.add_argument('--vocab_file', type=str, default="/home/dongqian06/codes/kgreranker/base/vocab.txt")
    parser.add_argument('--train_input_file', type=str,default="/home/dongqian06/hdfs_data/data_train/train.concept.tar.gz")
    parser.add_argument('--dev_input_file', type=str, default="/home/dongqian06/data.tar.gz")
    parser.add_argument('--pretrain_input_file', type=str, default="/home/dongqian06/hdfs_data/data_train/pretrain/*")
    parser.add_argument('--pretrain_batch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dev_batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=160)
    parser.add_argument('--warm_start_from', type=str, default="/home/dongqian06/hdfs_data/data_train/ernie_base.p")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--eval_step_proportion', type=float, default=1.0)
    parser.add_argument('--ernie_lr', type=float, default=1e-5)
    parser.add_argument('--report', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--qrels', type=str, default="/home/dongqian06/hdfs_data/data_train/qrels.train.tsv")
    parser.add_argument('--top1000', type=str, default="/home/dongqian06/hdfs_data/data_train/train.concept.gz")
    parser.add_argument('--collection', type=str, default="/home/dongqian06/hdfs_data/data_train/collection.tsv")
    parser.add_argument('--query', type=str, default="/home/dongqian06/hdfs_data/data_train/train.query.txt")
    parser.add_argument('--min_index', type=int, default=25)
    parser.add_argument('--max_index', type=int, default=768)
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--fp16', type=bool, default=False)


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
    
    parser.add_argument('--resource', type=str, default="/home/dongqian06/hdfs_data/concept_net/concept.txt")
    parser.add_argument('--cpnet', type=str, default="/home/dongqian06/hdfs_data/concept_net/conceptnet.en.pruned.graph")
    parser.add_argument('--pattern_path', type=str, default="/home/dongqian06/hdfs_data/concept_net/matcher_patterns.json")
    parser.add_argument('--word2vec', type=str, default="/home/dongqian06/hdfs_data/data_train/GoogleNews-vectors-negative300.bin.gz")
    parser.add_argument('--topk_sents', type=int, default=1)
    parser.add_argument('--ent_emb', type=str, default="/home/dongqian06/hdfs_data/concept_net/glove.transe.sgd.ent.npy")
    parser.add_argument('--rel_emb', type=str, default="/home/dongqian06/hdfs_data/concept_net/glove.transe.sgd.rel.npy")
    parser.add_argument('--books', type=str, default="/home/dongqian06/hdfs_data/data_train/books.txt")
    parser.add_argument('--cnts', type=str, default="/home/dongqian06/hdfs_data/data_train/cnts.pkl")
    parser.add_argument('--gnn_hidden_size', type=int, default=100)
    parser.add_argument('--instance_num', type=int, default=109)# 502939
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args


    
def func(cfg):
    # cpu_worker_num = multiprocessing.cpu_count()
    cpu_worker_num = 26
    print(cpu_worker_num)
    local_rank = cfg['local_rank']
    books = open(cfg['books'],'r')
    books = books.readlines()
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe('sentencizer')
    def lemmatize(concept):
        doc = nlp(concept.replace("_", " "))
        lcs = set()
        lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
        return lcs
    def count(sent):
        sent = sent.lower()
        sent = sent.replace("-", "_")
        spans = []
        tokens = sent.split(" ")
        token_num = len(tokens)
        for length in range(1, 5):
            for i in range(token_num-length+1):
                span = "_".join(tokens[i:i+length])
                span = list(lemmatize(span))[0]
                if span in cnts:
                    cnts[span]+=1
                    # print(span)
    cnts = pkl.load(open(cfg['cnts'],"rb"))
    length = len(books)
    # length = 100
    process_args = [(i*length//cpu_worker_num, (i+1)*length//cpu_worker_num) for i in range(cpu_worker_num)]
    a,b = process_args[local_rank]
    local_start = time.time()
    for i in tqdm(range(a,b)):
        para = books[i]
        count(para)
    os.makedirs('output',exist_ok=True)
    pkl.dump(cnts, open("output/cnts_%d.pkl"%local_rank, "wb"))

if __name__=="__main__":
    args = define_args()
    all_configs = utils.parse_file(args.ernie_config_file)
    all_configs.update(vars(args))
    all_configs = utils.HParams(**all_configs)
    func(all_configs)
