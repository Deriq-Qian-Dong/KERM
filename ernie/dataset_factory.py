#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import json
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
class HardNegativeDataset(Dataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.data = pd.read_csv(cfg['train_input_file'],sep="\t",header=None)
        self.num_samples = len(self.data)
        self.gb = BatchedGraphBuilder(cfg)

    def _collate_fn(self, sample_list):
        batch = [np.stack(s, axis=0) for s in list(zip(*sample_list))]
        g_pos = self.gb.get_batched_graph(batch[0])
        g_neg = self.gb.get_batched_graph(batch[2])
        batch.append(g_pos)
        batch.append(g_neg)
        return batch

    def pairs_to_features(self, slot1, slot2):
        tokens_a,tokens_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return [input_ids, token_type_ids]

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        tokens_a = tokens_a[:]
        tokens_b = tokens_b[:]
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def convert_data_to_features(self, slot1, slot2, slot3):
        pos_features = self.pairs_to_features(slot1, slot2)
        neg_features = self.pairs_to_features(slot1, slot3)
        # features = src_ids, sent_ids, src_ids, sent_ids
        features = pos_features + neg_features
        return features

    def read_pos_neg_text(self, idx):
        cols = self.data.iloc[idx]
        text_a = cols[0]
        text_b = cols[2]
        text_c = cols[4]
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        tokens_c = self.tokenizer.tokenize(text_c)
        return tokens_a, tokens_b, tokens_c

    def __getitem__(self, idx):
        tokens_a, tokens_b, tokens_c = self.read_pos_neg_text(idx)
        # return src_ids, sent_ids, src_ids, sent_ids
        return self.convert_data_to_features(tokens_a, tokens_b, tokens_c)
        

    def __len__(self):
        return self.num_samples

class genDenoisedDataset(Dataset):
    def __init__(self, cfg):
        self.tokenizer = tokenization.FullTokenizer(cfg['vocab_file'])
        self.cfg = cfg
        self.collection = pd.read_csv(cfg['collection'],sep="\t",header=None)
        self.collection.columns=['pid','para']
        self.query = pd.read_csv(cfg['query'],sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(cfg['top1000'],sep="\t",header=None)
        self.top1000.columns=['qid','pid','index','score']
        self.num_samples = len(self.top1000)
        self.min_index = cfg['min_index']
        self.max_index = cfg['max_index']
        qrels={}
        with open(cfg['qrels'],'rb') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels

    def _collate_fn(self, sample_list):
        tmp = list(zip(*sample_list))
        batch = [np.stack(s, axis=0) for i,s in enumerate(tmp)]
        return batch

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        tokens_a = tokens_a[:]
        tokens_b = tokens_b[:]
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b     
    def pairs_to_features(self, slot1, slot2):
        tokens_a,tokens_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return [input_ids, token_type_ids]
    def convert_data_to_features(self, slot1, slot2):
        src_ids, sent_ids = self.pairs_to_features(slot1, slot2)
        # features = src_ids, sent_ids
        return src_ids, sent_ids

    def __getitem__(self, idx):
        cols = self.top1000.iloc[idx]
        qid = int(cols[0])
        pid = int(cols[1])
        text_a = self.query.loc[qid].text
        text_b = self.collection.iloc[pid]['para']
        src_ids, sent_ids = self.convert_data_to_features(self.tokenizer.tokenize(text_a), self.tokenizer.tokenize(text_b))
        return src_ids, sent_ids,qid,pid
        

    def __len__(self):
        return self.num_samples


class TrainDataset(Dataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.gb = BatchedGraphBuilder(cfg)
        self.collection = pd.read_csv(cfg['collection'],sep="\t",header=None)
        self.collection.columns=['pid','para']
        self.query = pd.read_csv(cfg['query'],sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(cfg['top1000'],sep="\t",header=None)
        self.top1000.columns=['qid','pid','index','score']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.num_samples = len(self.top1000)
        self.min_index = cfg['min_index']
        self.max_index = cfg['max_index']
        qrels={}
        with open(cfg['qrels'],'rb') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels


    def _collate_fn(self, sample_list):
        batch = [np.stack(s, axis=0) for s in list(zip(*sample_list))]
        g_pos = self.gb.get_batched_graph(batch[0])
        g_neg = self.gb.get_batched_graph(batch[2])
        batch.append(g_pos)
        batch.append(g_neg)
        return batch

    def pairs_to_features(self, slot1, slot2):
        tokens_a,tokens_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return [input_ids, token_type_ids]

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        tokens_a = tokens_a[:]
        tokens_b = tokens_b[:]
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def convert_data_to_features(self, slot1, slot2, slot3):
        pos_features = self.pairs_to_features(slot1, slot2)
        neg_features = self.pairs_to_features(slot1, slot3)
        # features = src_ids, sent_ids, src_ids, sent_ids
        features = pos_features + neg_features
        return features

    def read_pos_neg_text(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pos_id = random.choice(self.qrels.get(qid))
        found = False
        for i in range(1000):
            neg_id = random.choice(list(cols[1]['pid'])[self.min_index:self.max_index])
            if neg_id not in self.qrels[qid]:
                found=True
                break
        if not found:
            text_c = "#"
        else:
            text_c = self.collection.iloc[neg_id]['para']
        text_a = self.query.loc[qid]['text']
        text_b = self.collection.iloc[pos_id]['para']
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        tokens_c = self.tokenizer.tokenize(text_c)
        return tokens_a, tokens_b, tokens_c

    def __getitem__(self, idx):
        tokens_a, tokens_b, tokens_c = self.read_pos_neg_text(idx)
        # return src_ids, sent_ids, src_ids, sent_ids
        return self.convert_data_to_features(tokens_a, tokens_b, tokens_c)
        

    def __len__(self):
        return self.num_samples


class TrainNCEDataset(Dataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.gb = BatchedGraphBuilder(cfg)
        self.collection = pd.read_csv(cfg['collection'],sep="\t",header=None)
        self.collection.columns=['pid','para']
        self.query = pd.read_csv(cfg['query'],sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(cfg['top1000'],sep="\t",header=None)
        self.top1000.columns=['qid','pid','index','score']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.num_samples = len(self.top1000)
        self.min_index = cfg['min_index']
        self.max_index = cfg['max_index']
        qrels={}
        with open(cfg['qrels'],'rb') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = cfg['sample_num']-1
        
    def _collate_fn(self, sample_list):
        batch = [np.concatenate(s, axis=0) for s in list(zip(*sample_list))[:-2]]
        g = self.gb.get_batched_graph(batch[0])
        batch+=[np.stack(s, axis=0) for s in list(zip(*sample_list))[-2:]]
        batch.append(g)
        return batch

    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        sample_pids = []
        if len(pids)<sample_num:
            sample_pids = [pid for pid in pids if pid not in self.qrels[qid]]
            sample_pids+=[0]*(sample_num - len(sample_pids))
            return sample_pids
        interval = len(pids)//sample_num
        for i in range(sample_num):
            found = False
            for _ in range(interval):
                neg_id = random.choice(pids[i*interval:(i+1)*interval])
                if neg_id not in self.qrels[qid]:
                    found = True
                    break
            if not found:
                neg_id = 0
            sample_pids.append(neg_id)
        return sample_pids
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        tokens_a = tokens_a[:]
        tokens_b = tokens_b[:]
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b     
    def pairs_to_features(self, slot1, slot2):
        tokens_a,tokens_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return [input_ids, token_type_ids]
    def convert_data_to_features(self, slot1, slot2):
        features = self.pairs_to_features(slot1, slot2)
        # features = src_ids, sent_ids
        return features

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid,pids,self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        query = self.tokenizer.tokenize(query)
        features = []
        features.append(self.convert_data_to_features(query, self.tokenizer.tokenize(self.collection.iloc[pos_id]['para'])))
        for neg_pid in sample_neg_pids:
            features.append(self.convert_data_to_features(query, self.tokenizer.tokenize(self.collection.iloc[neg_pid]['para'])))
        labels = [0]
        data = [np.stack(s, axis=0) for s in list(zip(*features))]
        data.append(labels)
        data.append([qid]*self.cfg['sample_num'])
        return data

    def __len__(self):
        return self.num_samples


class EvalDataset(Dataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.data = pd.read_csv(cfg['dev_input_file'],sep="\t",header=None)
        self.num_samples = len(self.data)
        self.gb = BatchedGraphBuilder(cfg)
        
    def _collate_fn(self, sample_list):
        batch = [np.stack(s, axis=0) for s in list(zip(*sample_list))]
        g = self.gb.get_batched_graph(batch[3])
        batch.append(g)
        return batch

    def pairs_to_features(self, slot1, slot2):
        tokens_a,tokens_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return [input_ids, token_type_ids]

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        tokens_a = tokens_a[:]
        tokens_b = tokens_b[:]
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def convert_data_to_features(self, slot1, slot2):
        features = self.pairs_to_features(slot1, slot2)
        # features = src_ids, sent_ids
        return features

    def read_line(self, idx):
        cols = self.data.iloc[idx]
        qid = int(cols[0])
        pid = int(cols[1])
        text_a = cols[2]
        text_b = cols[4]
        label = int(cols[5])
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        return qid, pid, tokens_a, tokens_b, label

    def __getitem__(self, idx):
        qid, pid, tokens_a, tokens_b, label = self.read_line(idx)
        pair_input_ids, pair_token_type_ids = self.convert_data_to_features(tokens_a, tokens_b)
        # return src_ids, sent_ids, src_ids, sent_ids
        return  qid, pid, label, pair_input_ids, pair_token_type_ids

        

    def __len__(self):
        return self.num_samples

class GenTrainErnieConceptDataset(Dataset):
    def __init__(self, cfg):
        self.tokenizer = tokenization.FullTokenizer(cfg['vocab_file'])
        self.cfg = cfg
        self.gb = ConceptGraphBuilder(cfg)
        self.collection = pd.read_csv(cfg['collection'],sep="\t",header=None, quoting=3)
        self.collection.columns=['pid','para']
        self.collection = self.collection.fillna("NA")
        self.query = pd.read_csv(cfg['query'],sep="\t",header=None)
        self.query.columns = ['qid','text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(cfg['top1000'],sep="\t",header=None)
        self.top1000.columns=['qid','pid','index','score']
        self.top1000 = list(self.top1000.groupby("qid"))
        self.num_samples = len(self.top1000)
        self.min_index = cfg['min_index']
        self.max_index = cfg['max_index']
        qrels={}
        with open(cfg['qrels'],'r') as f:
            lines = f.readlines()
            for line in lines:
                qid,pid = line.split()
                qid=int(qid)
                pid=int(pid)
                x=qrels.get(qid,[])
                x.append(pid)
                qrels[qid]=x
        self.qrels = qrels
        self.sample_num = cfg['sample_num']-1
    # def _collate_fn(sample_list):
    #     '''
    #         data = [np.stack(s, axis=0) for s in list(zip(*features))]
    #         data.append(q_spans)
    #         data.append(d_spans)
    #         data.append(q_nodes)
    #         data.append(d_nodes)
    #         data.append(graphs)
    #         data.append(labels)
    #         data.append([qid]*self.cfg['sample_num'])
    #     '''
    #     batch = [np.concatenate(s, axis=0) for s in list(zip(*sample_list))[:-5]]
    #     graphs = []
    #     for g in list(zip(*sample_list))[-5]:
    #         graphs+=g
    #     q_spans = []
    #     for s in list(zip(*sample_list))[-4]:
    #         q_spans+=s
    #     d_spans = []
    #     for s in list(zip(*sample_list))[-3]:
    #         d_spans+=s
    #     q_nodes = []
    #     for n in list(zip(*sample_list))[-2]:
    #         q_nodes+=n
    #     d_nodes = []
    #     for n in list(zip(*sample_list))[-1]:
    #         d_nodes+=n
    #     batch.append(pgl.Graph.batch(graphs))
    #     batch.append(q_spans)
    #     batch.append(d_spans)
    #     batch.append(q_nodes)
    #     batch.append(d_nodes)
    #     return batch
    def _collate_fn(self, sample_list):
        '''
            data = [np.stack(s, axis=0) for s in list(zip(*features))]
            data.append(q_spans)
            data.append(d_spans)
            data.append(q_nodes)
            data.append(d_nodes)
            data.append(graphs)
            data.append(labels)
            data.append([qid]*self.cfg['sample_num'])
        '''
        batch = [np.concatenate(s, axis=0) for s in list(zip(*sample_list))[:-7]]
        batch+=[np.stack(s, axis=0) for s in list(zip(*sample_list))[-2:]]
        graphs = []
        for g in list(zip(*sample_list))[-3]:
            graphs+=g
        q_spans = []
        for s in list(zip(*sample_list))[-7]:
            q_spans+=s
        d_spans = []
        for s in list(zip(*sample_list))[-6]:
            d_spans+=s
        q_nodes = []
        for n in list(zip(*sample_list))[-5]:
            q_nodes+=n
        d_nodes = []
        for n in list(zip(*sample_list))[-4]:
            d_nodes+=n
        batch.append(pgl.Graph.batch(graphs))
        batch.append(q_spans)
        batch.append(d_spans)
        batch.append(q_nodes)
        batch.append(d_nodes)
        return batch
    def _collate_fn_gen(self, sample_list):
        '''
            data = [np.stack(s, axis=0) for s in list(zip(*features))]
            data.append(qry_mps)
            data.append(doc_mps)
            data.append(q_spans)
            data.append(d_spans)
            data.append(q_nodes)
            data.append(d_nodes)
            data.append(graphs)
            data.append(labels)
            data.append([qid]*self.cfg['sample_num'])
        '''
        tmp = list(zip(*sample_list))
        batch = [np.concatenate(s, axis=0) for s in tmp[:-9]]
        batch+=[np.stack(s, axis=0) for s in tmp[-2:]]
        q_mps = tmp[-9][0]
        d_mps = tmp[-8][0]
        q_spans = tmp[-7][0]
        d_spans = tmp[-6][0]
        q_nodes = tmp[-5][0]
        d_nodes = tmp[-4][0]
        graphs = tmp[-3][0]
        
        # batch.append(pgl.Graph.batch(graphs))
        batch.append(graphs)
        batch.append(q_spans)
        batch.append(d_spans)
        batch.append(q_nodes)
        batch.append(d_nodes)
        batch.append(q_mps)
        batch.append(d_mps)
        return batch
    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        sample_pids = []
        # pids = pids[self.cfg['min_index']:self.cfg['max_index']]
        if len(pids)<sample_num:
            sample_pids = [pid for pid in pids if pid not in self.qrels[qid]]
            sample_pids+=[1]*(sample_num - len(sample_pids))
            return sample_pids
        interval = len(pids)//sample_num
        for i in range(sample_num):
            found = False
            for _ in range(interval):
                neg_id = random.choice(pids[i*interval:(i+1)*interval])
                if neg_id not in self.qrels[qid]:
                    found = True
                    break
            if not found:
                neg_id = 1
            sample_pids.append(neg_id)
        return sample_pids
    def tokenize_with_align(self, text):
        mp = {}
        toks=[]
        # text = text.lower()
        # text = re.sub(r'[^a-z ]+', '', text)
        text = text.split(" ")
        for i,t in enumerate(text):
            mp[i] = len(toks)
            toks+=self.tokenizer.tokenize(t)
        return toks, mp
    def _truncate_seq_pair(self, text_a, text_b, max_length):
        tokens_a, mp_a = self.tokenize_with_align(text_a)
        tokens_b, mp_b = self.tokenize_with_align(text_b)
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        mp_a = dict((key, value) for key, value in mp_a.items() if value <= len(tokens_a))
        mp_b = dict((key, value) for key, value in mp_b.items() if value <= len(tokens_b))
        return tokens_a, tokens_b, mp_a, mp_b
    def convert_data_to_features(self, slot1, slot2):
        tokens_a, tokens_b, mp_a, mp_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        qry = 'cccc ' + slot1 
        doc = 'ssss ' + slot2 
        mp_a = dict((key+1, 1+value) for key, value in mp_a.items())
        mp_a[0]=0
        mp_b = dict((key+1, 1+len(tokens_a)+1+value) for key, value in mp_b.items())
        mp_b[0] = 1+len(tokens_a)
        graph,q_spans,d_spans,qry_nodes,doc_nodes = self.gb.get_graph(qry, doc,False)
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return [input_ids, token_type_ids], graph, q_spans, d_spans, qry_nodes,doc_nodes, mp_a, mp_b

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid,pids,self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        features = []
        graphs = []
        q_spans = []
        d_spans = []
        q_nodes = []
        d_nodes = []
        qry_mps = []
        doc_mps = []
        feature, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp = self.convert_data_to_features(query, self.collection.iloc[pos_id]['para'])
        q_spans.append(q_span)
        q_nodes.append(qry_node)
        d_spans.append(d_span)
        d_nodes.append(doc_node)
        features.append(feature)
        graphs.append(graph)
        qry_mps.append(qry_mp)
        doc_mps.append(doc_mp)
        for neg_pid in sample_neg_pids:
            feature, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp = self.convert_data_to_features(query, self.collection.iloc[neg_pid]['para'])
            q_spans.append(q_span)
            q_nodes.append(qry_node)
            d_spans.append(d_span)
            d_nodes.append(doc_node)
            features.append(feature)
            qry_mps.append(qry_mp)
            doc_mps.append(doc_mp)
            graphs.append(graph)
        labels = [0]
        data = [np.stack(s, axis=0) for s in list(zip(*features))]
        data.append(qry_mps)
        data.append(doc_mps)
        data.append(q_spans)
        data.append(d_spans)
        data.append(q_nodes)
        data.append(d_nodes)
        data.append(graphs)
        data.append(labels)
        data.append([qid]*self.cfg['sample_num'])
        return data

    def __len__(self):
        return self.num_samples

from collections import namedtuple
def csv_reader(fd, delimiter='\t'):
    def gen():
        for i in fd:
            yield i.rstrip('\n').split(delimiter)
    return gen()
def _read_tsv(input_file, batch_size=16, quotechar=None):
    """Reads a tab separated value file."""
    
    with open(input_file, 'r', encoding='utf8') as f:
        reader = csv_reader(f)
        #headers = next(reader)
        headers = 'query\ttitle\tpara\tlabel'.split('\t')
        text_indices = [
            index for index, h in enumerate(headers) if h != "label"
        ]
        Example = namedtuple('Example', headers)

        examples = []
        for line in reader:
            for index, text in enumerate(line):
                if index in text_indices:
                    line[index] = text
            example = Example(*line)
            examples.append(example)
        while len(examples) % batch_size != 0:
            examples.append(example)
        return examples
class GenTrainV2ErnieConceptDataset(Dataset):
    def __init__(self, cfg):
        self.tokenizer = tokenization.FullTokenizer(cfg['vocab_file'])
        self.cfg = cfg
        self.gb = ConceptGraphBuilder(cfg)
        reader = _read_tsv(cfg['top1000'])
        query2cands={}
        for example in reader:
            x = query2cands.get(example.query,{"pos":[],"neg":[]})
            if example.label=="1":
                x["pos"].append(example)
            else:
                x["neg"].append(example)
            query2cands[example.query] = x
        self.top1000 = []
        for qry in query2cands:
            for pos in query2cands[qry]['pos']:
                self.top1000.append([pos,query2cands[qry]['neg']])
        self.num_samples = len(self.top1000)
        self.min_index = cfg['min_index']
        self.max_index = cfg['max_index']
        self.sample_num = cfg['sample_num']-1
    def _collate_fn(self, sample_list):
        '''
            data = [np.stack(s, axis=0) for s in list(zip(*features))]
            data.append(q_spans)
            data.append(d_spans)
            data.append(q_nodes)
            data.append(d_nodes)
            data.append(graphs)
            data.append(labels)
            data.append([qid]*self.cfg['sample_num'])
        '''
        batch = [np.concatenate(s, axis=0) for s in list(zip(*sample_list))[:-7]]
        batch+=[np.stack(s, axis=0) for s in list(zip(*sample_list))[-2:]]
        graphs = []
        for g in list(zip(*sample_list))[-3]:
            graphs+=g
        q_spans = []
        for s in list(zip(*sample_list))[-7]:
            q_spans+=s
        d_spans = []
        for s in list(zip(*sample_list))[-6]:
            d_spans+=s
        q_nodes = []
        for n in list(zip(*sample_list))[-5]:
            q_nodes+=n
        d_nodes = []
        for n in list(zip(*sample_list))[-4]:
            d_nodes+=n
        batch.append(pgl.Graph.batch(graphs))
        batch.append(q_spans)
        batch.append(d_spans)
        batch.append(q_nodes)
        batch.append(d_nodes)
        return batch
    def _collate_fn_gen(self, sample_list):
        '''
            data = [np.stack(s, axis=0) for s in list(zip(*features))]
            data.append(qry_mps)
            data.append(doc_mps)
            data.append(q_spans)
            data.append(d_spans)
            data.append(q_nodes)
            data.append(d_nodes)
            data.append(graphs)
            data.append(labels)
            data.append([qid]*self.cfg['sample_num'])
        '''
        tmp = list(zip(*sample_list))
        batch = [np.concatenate(s, axis=0) for s in tmp[:-9]]
        batch+=[np.stack(s, axis=0) for s in tmp[-2:]]
        q_mps = tmp[-9][0]
        d_mps = tmp[-8][0]
        q_spans = tmp[-7][0]
        d_spans = tmp[-6][0]
        q_nodes = tmp[-5][0]
        d_nodes = tmp[-4][0]
        graphs = tmp[-3][0]
        
        # batch.append(pgl.Graph.batch(graphs))
        batch.append(graphs)
        batch.append(q_spans)
        batch.append(d_spans)
        batch.append(q_nodes)
        batch.append(d_nodes)
        batch.append(q_mps)
        batch.append(d_mps)
        return batch
    def sample(self, negs, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        sample_negs = []
        interval = len(negs)//sample_num
        for i in range(sample_num):
            neg = random.choice(negs[i*interval:(i+1)*interval])
            sample_negs.append(neg)
        return sample_negs
    def tokenize_with_align(self, text):
        mp = {}
        toks=[]
        # text = text.lower()
        # text = re.sub(r'[^a-z ]+', '', text)
        text = text.split(" ")
        for i,t in enumerate(text):
            mp[i] = len(toks)
            toks+=self.tokenizer.tokenize(t)
        return toks, mp
    def _truncate_seq_pair(self, text_a, text_b, max_length):
        tokens_a, mp_a = self.tokenize_with_align(text_a)
        tokens_b, mp_b = self.tokenize_with_align(text_b)
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        mp_a = dict((key, value) for key, value in mp_a.items() if value <= len(tokens_a))
        mp_b = dict((key, value) for key, value in mp_b.items() if value <= len(tokens_b))
        return tokens_a, tokens_b, mp_a, mp_b
    def convert_data_to_features(self, slot1, slot2):
        tokens_a, tokens_b, mp_a, mp_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        qry = 'cccc ' + slot1 
        doc = 'ssss ' + slot2 
        mp_a = dict((key+1, 1+value) for key, value in mp_a.items())
        mp_a[0]=0
        mp_b = dict((key+1, 1+len(tokens_a)+1+value) for key, value in mp_b.items())
        mp_b[0] = 1+len(tokens_a)
        graph,q_spans,d_spans,qry_nodes,doc_nodes = self.gb.get_graph(qry, doc,False)
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return [input_ids, token_type_ids], graph, q_spans, d_spans, qry_nodes,doc_nodes, mp_a, mp_b

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        pos = cols[0]
        negs = cols[1]
        sample_negs = self.sample(negs, self.sample_num)
        query = pos.query
        features = []
        graphs = []
        q_spans = []
        d_spans = []
        q_nodes = []
        d_nodes = []
        qry_mps = []
        doc_mps = []
        feature, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp = self.convert_data_to_features(query, pos.title+" "+pos.para)
        q_spans.append(q_span)
        q_nodes.append(qry_node)
        d_spans.append(d_span)
        d_nodes.append(doc_node)
        features.append(feature)
        graphs.append(graph)
        qry_mps.append(qry_mp)
        doc_mps.append(doc_mp)
        for neg in sample_negs:
            feature, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp = self.convert_data_to_features(query, neg.title+" "+neg.para)
            q_spans.append(q_span)
            q_nodes.append(qry_node)
            d_spans.append(d_span)
            d_nodes.append(doc_node)
            features.append(feature)
            qry_mps.append(qry_mp)
            doc_mps.append(doc_mp)
            graphs.append(graph)
        labels = [0]
        data = [np.stack(s, axis=0) for s in list(zip(*features))]
        data.append(qry_mps)
        data.append(doc_mps)
        data.append(q_spans)
        data.append(d_spans)
        data.append(q_nodes)
        data.append(d_nodes)
        data.append(graphs)
        data.append(labels)
        data.append([idx]*self.cfg['sample_num'])
        return data

    def __len__(self):
        return self.num_samples

class GenEvalErnieConceptDataset(Dataset):
    def __init__(self, cfg):
        self.tokenizer = tokenization.FullTokenizer(cfg['vocab_file'])
        self.cfg = cfg
        self.data = pd.read_csv(cfg['dev_input_file'],sep="\t",header=None)
        self.gb = ConceptGraphBuilder(cfg)
        # self.run = pd.read_csv('data/run.bm25.dev.small.tsv',sep="\t",header=None)
        self.num_samples = len(self.data)
        self.is_test = False
        # self.collection = pd.read_csv(cfg['collection'],sep="\t",header=None)
        # self.collection.columns=['pid','para']
        # self.query = pd.read_csv('data/dev.query.txt',sep="\t",header=None)
        # self.query.columns = ['qid','text']
        # self.query.index = self.query.qid
        # self.query.pop('qid')
        # qrels={}
        # with open("data/qrels.dev.tsv",'rb') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         qid,pid = line.split()
        #         qid=int(qid)
        #         pid=int(pid)
        #         x=qrels.get(qid,[])
        #         x.append(pid)
        #         qrels[qid]=x
        # self.qrels = qrels
    def _collate_fn_gen(self, sample_list):
        '''
        qid, pid, label, input_ids, token_type_ids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp
        '''
        tmp = list(zip(*sample_list))
        batch = [np.stack(s, axis=0) for s in tmp[:5]]
        batch+=tmp[5:]
        return batch 
    def tokenize_with_align(self, text):
        mp = {}
        toks=[]
        # text = text.lower()
        # text = re.sub(r'[^a-z ]+', '', text)
        text = text.split(" ")
        for i,t in enumerate(text):
            mp[i] = len(toks)
            toks+=self.tokenizer.tokenize(t)
        return toks, mp
    def _truncate_seq_pair(self, text_a, text_b, max_length):
        tokens_a, mp_a = self.tokenize_with_align(text_a)
        tokens_b, mp_b = self.tokenize_with_align(text_b)
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        mp_a = dict((key, value) for key, value in mp_a.items() if value <= len(tokens_a))
        mp_b = dict((key, value) for key, value in mp_b.items() if value <= len(tokens_b))
        return tokens_a, tokens_b, mp_a, mp_b
    def convert_data_to_features(self, slot1, slot2):
        tokens_a, tokens_b, mp_a, mp_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        qry = 'cccc ' + slot1 
        doc = 'ssss ' + slot2 
        mp_a = dict((key+1, 1+value) for key, value in mp_a.items())
        mp_a[0]=0
        mp_b = dict((key+1, 1+len(tokens_a)+1+value) for key, value in mp_b.items())
        mp_b[0] = 1+len(tokens_a)
        graph,q_spans,d_spans,qry_nodes,doc_nodes = self.gb.get_graph(qry, doc, False)
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return input_ids, token_type_ids, graph, q_spans, d_spans, qry_nodes,doc_nodes, mp_a, mp_b

    def __getitem__(self, idx):
        cols = self.data.iloc[idx]
        # cols = self.run.iloc[idx]
        qid = int(cols[0])
        pid = int(cols[1])
        text_a = cols[2]
        if self.is_test:
            text_b = cols[3]
        else:
            text_b = cols[4]
        # text_a = self.query.loc[qid].text
        # text_b = self.collection.iloc[pid]['para']
        if self.is_test:
            label = 0
        else:
            label = int(cols[5])
        # label = 1 if pid in self.qrels.get(qid, []) else 0
        input_ids, token_type_ids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp = self.convert_data_to_features(text_a, text_b)
        return qid, pid, label, input_ids, token_type_ids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp

    def __len__(self):
        return self.num_samples

class GenPretrainedConceptDataset(Dataset):
    def __init__(self, cfg, start, end):
        self.tokenizer = tokenization.FullTokenizer(cfg['vocab_file'])
        self.cfg = cfg
        self.data = pd.read_csv(cfg['collection'],header=None,sep='\t')
        self.data.columns=['pid','text']
        self.data = self.data.iloc[start:end]
        self.num_samples = len(self.data)
        self.gb = ConceptGraphBuilder(cfg)
        self.nlp = self.gb.nlp
        self.cnts = [0,0,0]

    def _collate_fn_gen(self, sample_list):
        '''
        qid, pid, label, input_ids, token_type_ids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp
        '''
        return sample_list 
    def tokenize_with_align(self, text):
        mp = {}
        toks=[]
        # text = text.lower()
        # text = re.sub(r'[^a-z ]+', '', text)
        text = text.split(" ")
        for i,t in enumerate(text):
            mp[i] = len(toks)
            toks+=self.tokenizer.tokenize(t)
        return toks, mp
    def _truncate_seq_pair(self, text_a, text_b, max_length):
        tokens_a, mp_a = self.tokenize_with_align(text_a)
        tokens_b, mp_b = self.tokenize_with_align(text_b)
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        mp_a = dict((key, value) for key, value in mp_a.items() if value <= len(tokens_a))
        mp_b = dict((key, value) for key, value in mp_b.items() if value <= len(tokens_b))
        return tokens_a, tokens_b, mp_a, mp_b
    def convert_data_to_features(self, slot1, slot2):
        tokens_a, tokens_b, mp_a, mp_b = self._truncate_seq_pair(slot1, slot2, self.cfg['max_seq_len']-3)
        slot = ['[CLS]'] + tokens_a+ ['[SEP]'] + tokens_b + ['[SEP]']
        qry = 'cccc ' + slot1 
        doc = 'ssss ' + slot2 
        mp_a = dict((key+1, 1+value) for key, value in mp_a.items())
        mp_a[0]=0
        mp_b = dict((key+1, 1+len(tokens_a)+1+value) for key, value in mp_b.items())
        mp_b[0] = 1+len(tokens_a)
        graph,q_spans,d_spans,qry_nodes,doc_nodes = self.gb.get_graph(qry, doc, False)
        input_ids = self.tokenizer.convert_tokens_to_ids(slot)
        input_ids = np.array(input_ids, dtype=np.int64)
        token_type_ids = np.array([0]*(2+len(tokens_a))+[1]*(1+len(tokens_b)),dtype=np.int64)
        input_ids = np.pad(input_ids, (0, self.cfg['max_seq_len']-len(input_ids)), 'constant', constant_values=(0,0))
        token_type_ids = np.pad(token_type_ids, (0, self.cfg['max_seq_len']-len(token_type_ids)), 'constant', constant_values=(0,0))
        return input_ids, token_type_ids, graph, q_spans, d_spans, qry_nodes,doc_nodes, mp_a, mp_b

    def __getitem__(self, idx):
        cols = self.data.iloc[idx]
        text = cols.text
        doc = self.nlp(text)
        sents = list(doc.sents)
        sent_number = len(sents)
        input_ids_list = []
        token_type_ids_list = []
        graph_list, q_span_list, d_span_list, qry_node_list, doc_node_list, qry_mp_list, doc_mp_list = [],[],[],[],[],[],[]
        labels = []
        for j, sent in enumerate(sents):
            text_a = sent.text
            if j and j<sent_number-1:
                srp_type = self.cnts.index(min(self.cnts))
                self.cnts[srp_type]+=1
                if srp_type==0:#random
                    sampled_doc_id = random.randint(0, self.num_samples-1)
                    sampled_cols = self.data.iloc[sampled_doc_id]
                    sampled_doc = self.nlp(sampled_cols.text)
                    sampled_sents = list(sampled_doc.sents)
                    text_b = random.choice(sampled_sents).text
                elif srp_type==1:#next
                    text_b = sents[j+1].text
                else:#previous
                    text_b = sents[j-1].text
            elif j and j==sent_number-1:
                srp_type = 0 if self.cnts[0]<self.cnts[2] else 2
                self.cnts[srp_type]+=1
                if srp_type==0:
                    sampled_doc_id = random.randint(0, self.num_samples-1)
                    sampled_cols = self.data.iloc[sampled_doc_id]
                    sampled_doc = self.nlp(sampled_cols.text)
                    sampled_sents = list(sampled_doc.sents)
                    text_b = random.choice(sampled_sents).text
                elif srp_type==2:
                    text_b = sents[j-1].text
            elif j==0 and sent_number>1:
                srp_type = 0 if self.cnts[0]<self.cnts[1] else 1
                self.cnts[srp_type]+=1
                if srp_type==0:
                    sampled_doc_id = random.randint(0, self.num_samples-1)
                    sampled_cols = self.data.iloc[sampled_doc_id]
                    sampled_doc = self.nlp(sampled_cols.text)
                    sampled_sents = list(sampled_doc.sents)
                    text_b = random.choice(sampled_sents).text
                elif srp_type==1:
                    text_b = sents[j+1].text
            elif j==0 and sent_number==1:
                srp_type = 0
                self.cnts[srp_type]+=1
                sampled_doc_id = random.randint(0, self.num_samples-1)
                sampled_cols = self.data.iloc[sampled_doc_id]
                sampled_doc = self.nlp(sampled_cols.text)
                sampled_sents = list(sampled_doc.sents)
                text_b = random.choice(sampled_sents).text
            input_ids, token_type_ids, graph, q_span, d_span, qry_node, doc_node, qry_mp, doc_mp = self.convert_data_to_features(text_a, text_b)
            labels.append(srp_type)
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            graph_list.append(graph)
            q_span_list.append(q_span)
            d_span_list.append(d_span)
            qry_node_list.append(qry_node)
            doc_node_list.append(doc_node)
            qry_mp_list.append(qry_mp)
            doc_mp_list.append(doc_mp)
        return labels, input_ids_list, token_type_ids_list, graph_list, q_span_list, d_span_list, qry_node_list, doc_node_list, qry_mp_list, doc_mp_list
    def __len__(self):
        return self.num_samples


class TrainErnieConceptDataset(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_procs = dist.get_world_size()  # 8
        self.rank = dist.get_rank()  # rank
        self.data = tarfile.open(self.cfg['train_input_file'])
        self.batch_size = cfg['batch_size']
    def sample(self, sample_num):
        '''
        sample_num:int
        return:sample_idx
        '''
        sample_idx = []
        pids = list(range(1,self.cfg['sample_range']))
        interval = len(pids)//sample_num
        for i in range(sample_num):
            neg_id = random.choice(pids[i*interval:(i+1)*interval])
            sample_idx.append(neg_id)
        return sample_idx
    @property
    def get_batch(self):
        batch = []
        for sample in self.data:
            if 'sample' in sample.get_info()['name']:
                sample=pkl.load(self.data.extractfile(sample))
                sample_idx = [0] + self.sample(self.cfg['sample_num']-1)
                input_ids, token_type_ids, label, qids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp=sample
                input_ids = input_ids[sample_idx]
                token_type_ids = token_type_ids[sample_idx]
                qids = np.array([qids[0][idx] for idx in sample_idx]).reshape((1,-1))
                graph = [graph[idx] for idx in sample_idx]
                q_span = [q_span[idx] for idx in sample_idx]
                d_span = [d_span[idx] for idx in sample_idx]
                qry_node = [qry_node[idx] for idx in sample_idx]
                doc_node = [doc_node[idx] for idx in sample_idx]
                qry_mp = [qry_mp[idx] for idx in sample_idx]
                doc_mp = [doc_mp[idx] for idx in sample_idx]
                sample = (input_ids, token_type_ids, label, qids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp)
                batch.append(sample)
                if len(batch)==self.batch_size*self.n_procs:
                    yield batch
                    batch = []
        if len(batch) and len(batch)<self.batch_size*self.n_procs:
            batch*=self.batch_size*self.n_procs
            batch = batch[:self.batch_size*self.n_procs]
            yield batch
            batch = []
    def __iter__(self):
        for batch in self.get_batch:
            batch = batch[self.rank::self.n_procs]
            batch = self._collate_fn(batch)
            yield batch
    def batched_graph(self, graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span):
        '''
        graphs:包含40*4个graph的list
        return:一个大graph，前160*40*4个node和ernie对应，后面的node是知识图谱中的
        '''
        node_num = len(graphs)*self.cfg['max_seq_len']
        node_feats = np.zeros((node_num,100))
        edges = []
        edges_feature = []
        concept_feats = []
        for i,graph in enumerate(graphs):
            node_feat = graph.node_feat['feature']
            edge_feat = graph.edge_feat['edge_feature']
            nodes = {}
            edge_feat = edge_feat.reshape((-1,100))
            edges_feature.append(edge_feat)
            qry_node = qrys_node[i]
            doc_node = docs_node[i]
            qry_node = {n:idx for idx,n in enumerate(qry_node)}
            doc_node = {n:idx for idx,n in enumerate(doc_node)}
            qry_mp = qrys_mp[i]
            doc_mp = docs_mp[i]
            qry_span = qrys_span[i]
            doc_span = docs_span[i]
            for idx,n in enumerate(qry_node):
                word_idx = qry_span[idx][0]
                ernie_idx = qry_mp.get(word_idx, -1)
                if ernie_idx!=-1:
                    node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                    nodes[n] = i*self.cfg['max_seq_len']+ernie_idx
            for idx,n in enumerate(doc_node):
                word_idx = doc_span[idx][0]
                ernie_idx = doc_mp.get(word_idx, -1)
                if ernie_idx!=-1:
                    node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                    nodes[n]=i*self.cfg['max_seq_len']+ernie_idx
            for source,target in graph.edges:
                new_source=0  #在大graph中的位置
                new_target=0  #在大graph中的位置
                if nodes.get(source, -1)!=-1:
                    new_source = nodes.get(source)
                else:
                    new_source=node_num
                    node_num+=1
                    concept_feats.append(node_feat[source])
                    nodes[source] = new_source
                ###################下面是target#######################
                if nodes.get(target,-1)!=-1:
                    new_target = nodes.get(target)
                else:
                    new_target=node_num
                    node_num+=1
                    concept_feats.append(node_feat[target])
                    nodes[target]=new_target
                edges.append((new_source,new_target))
        if len(concept_feats):
            concept_feats=np.stack(concept_feats,axis=0)
            nodes_feature=np.concatenate([node_feats,concept_feats],axis=0)
        else:
            nodes_feature = node_feats
        edges_feature=np.concatenate(edges_feature,axis=0)
        graph = pgl.Graph(num_nodes=node_num,
                edges=edges,
                node_feat={"feature": nodes_feature},
                edge_feat={"edge_feature":edges_feature})
        return graph
    def _collate_fn(self, sample_list):
        '''
        input_ids, token_type_ids, label, qids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp
        '''
        tmp = list(zip(*sample_list))
        batch = [np.concatenate(s, axis=0) for s in tmp[:4]]
        graphs = []
        for g in tmp[4]:
            graphs+=g
        qrys_span = []
        for s in tmp[5]:
            qrys_span+=s
        docs_span = []
        for s in tmp[6]:
            docs_span+=s
        qrys_node = []
        for n in tmp[7]:
            qrys_node+=n
        docs_node = []
        for n in tmp[8]:
            docs_node+=n
        qrys_mp = []
        for m in tmp[9]:
            qrys_mp+=(m)
        docs_mp = []
        for m in tmp[10]:
            docs_mp+=(m)
        batched_graph = self.batched_graph(graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span)
        batch.append(batched_graph)
        return batch

class TrainErnieConceptWithMLMDataset(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_procs = dist.get_world_size()  # 8
        self.rank = dist.get_rank()  # rank
        self.data = tarfile.open(self.cfg['train_input_file'])
        self.batch_size = cfg['batch_size']
    def sample(self, sample_num):
        '''
        sample_num:int
        return:sample_idx
        '''
        sample_idx = []
        pids = list(range(1,self.cfg['sample_range']))
        interval = len(pids)//sample_num
        for i in range(sample_num):
            neg_id = random.choice(pids[i*interval:(i+1)*interval])
            sample_idx.append(neg_id)
        return sample_idx
    @property
    def get_batch(self):
        batch = []
        for sample in self.data:
            if 'sample' in sample.get_info()['name']:
                sample=pkl.load(self.data.extractfile(sample))
                sample_idx = [0] + self.sample(self.cfg['sample_num']-1)
                input_ids, token_type_ids, label, qids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp=sample
                input_ids = input_ids[sample_idx]
                token_type_ids = token_type_ids[sample_idx]
                qids = np.array([qids[0][idx] for idx in sample_idx]).reshape((1,-1))
                graph = [graph[idx] for idx in sample_idx]
                q_span = [q_span[idx] for idx in sample_idx]
                d_span = [d_span[idx] for idx in sample_idx]
                qry_node = [qry_node[idx] for idx in sample_idx]
                doc_node = [doc_node[idx] for idx in sample_idx]
                qry_mp = [qry_mp[idx] for idx in sample_idx]
                doc_mp = [doc_mp[idx] for idx in sample_idx]
                sample = (input_ids, token_type_ids, label, qids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp)
                batch.append(sample)
                if len(batch)==self.batch_size*self.n_procs:
                    yield batch
                    batch = []
        if len(batch) and len(batch)<self.batch_size*self.n_procs:
            batch*=self.batch_size*self.n_procs
            batch = batch[:self.batch_size*self.n_procs]
            yield batch
            batch = []
    def __iter__(self):
        for batch in self.get_batch:
            batch = batch[self.rank::self.n_procs]
            batch = self._collate_fn(batch)
            yield batch
    def batched_graph(self, graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span):
        '''
        graphs:包含40*4个graph的list
        return:一个大graph，前160*40*4个node和ernie对应，后面的node是知识图谱中的
        '''
        node_num = len(graphs)*self.cfg['max_seq_len']
        node_feats = np.zeros((node_num,100))
        edges = []
        edges_feature = []
        concept_feats = []
        for i,graph in enumerate(graphs):
            node_feat = graph.node_feat['feature']
            edge_feat = graph.edge_feat['edge_feature']
            nodes = {}
            edge_feat = edge_feat.reshape((-1,100))
            edges_feature.append(edge_feat)
            qry_node = qrys_node[i]
            doc_node = docs_node[i]
            qry_node = {n:idx for idx,n in enumerate(qry_node)}
            doc_node = {n:idx for idx,n in enumerate(doc_node)}
            qry_mp = qrys_mp[i]
            doc_mp = docs_mp[i]
            qry_span = qrys_span[i]
            doc_span = docs_span[i]
            for idx,n in enumerate(qry_node):
                word_idx = qry_span[idx][0]
                ernie_idx = qry_mp.get(word_idx, -1)
                if ernie_idx!=-1:
                    node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                    nodes[n] = i*self.cfg['max_seq_len']+ernie_idx
            for idx,n in enumerate(doc_node):
                word_idx = doc_span[idx][0]
                ernie_idx = doc_mp.get(word_idx, -1)
                if ernie_idx!=-1:
                    node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                    nodes[n]=i*self.cfg['max_seq_len']+ernie_idx
            for source,target in graph.edges:
                new_source=0  #在大graph中的位置
                new_target=0  #在大graph中的位置
                if nodes.get(source, -1)!=-1:
                    new_source = nodes.get(source)
                else:
                    new_source=node_num
                    node_num+=1
                    concept_feats.append(node_feat[source])
                    nodes[source] = new_source
                ###################下面是target#######################
                if nodes.get(target,-1)!=-1:
                    new_target = nodes.get(target)
                else:
                    new_target=node_num
                    node_num+=1
                    concept_feats.append(node_feat[target])
                    nodes[target]=new_target
                edges.append((new_source,new_target))
        if len(concept_feats):
            concept_feats=np.stack(concept_feats,axis=0)
            nodes_feature=np.concatenate([node_feats,concept_feats],axis=0)
        else:
            nodes_feature = node_feats
        edges_feature=np.concatenate(edges_feature,axis=0)
        graph = pgl.Graph(num_nodes=node_num,
                edges=edges,
                node_feat={"feature": nodes_feature},
                edge_feat={"edge_feature":edges_feature})
        return graph
    def _collate_fn(self, sample_list):
        '''
        input_ids, token_type_ids, label, qids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp
        '''
        tmp = list(zip(*sample_list))
        batch = [np.concatenate(s, axis=0) for s in tmp[:4]]
        graphs = []
        for g in tmp[4]:
            graphs+=g
        qrys_span = []
        for s in tmp[5]:
            qrys_span+=s
        docs_span = []
        for s in tmp[6]:
            docs_span+=s
        qrys_node = []
        for n in tmp[7]:
            qrys_node+=n
        docs_node = []
        for n in tmp[8]:
            docs_node+=n
        qrys_mp = []
        for m in tmp[9]:
            qrys_mp+=(m)
        docs_mp = []
        for m in tmp[10]:
            docs_mp+=(m)
        batched_graph = self.batched_graph(graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span)
        batch.append(batched_graph)
        input_ids, mask_pos, mask_label = apply_mask(batch[0])
        batch[0] = input_ids
        batch.append(mask_pos)
        batch.append(mask_label)
        return batch



class DevErnieConceptDataset(IterableDataset):
    def __init__(self, cfg, is_test=False):
        self.cfg = cfg
        self.n_procs = dist.get_world_size()  # 8
        self.rank = dist.get_rank()  # rank
        if is_test:
            self.data = tarfile.open(self.cfg['test_input_file'])
        else:
            self.data = tarfile.open(self.cfg['dev_input_file'])
    def __iter__(self):
        total_sample = 0
        pad = None
        num_yield = 0
        cnt = 0
        for sample in self.data:
            if 'sample' in sample.get_info()['name']:
                total_sample+=1
                if cnt%self.n_procs==self.rank:
                    sample=pkl.load(self.data.extractfile(sample))
                    pad = sample
                    num_yield+=1
                    yield sample
            cnt+=1
        if total_sample%self.n_procs and total_sample//self.n_procs==num_yield:
            num_yield+=1
            qid, pid, label, input_ids, token_type_ids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp=pad
            qid = -100000000*np.ones_like(qid)
            pad = qid, pid, label, input_ids, token_type_ids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp
            yield pad
        
    def batched_graph(self, graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span):
        '''
        graphs:包含40*4个graph的list
        return:一个大graph，前160*40*4个node和ernie对应，后面的node是知识图谱中的
        '''
        node_num = len(graphs)*self.cfg['max_seq_len']
        node_feats = np.zeros((node_num,100))
        edges = []
        edges_feature = []
        concept_feats = []
        for i,graph in enumerate(graphs):
            node_feat = graph.node_feat['feature']
            edge_feat = graph.edge_feat['edge_feature']
            nodes = {}
            edge_feat = edge_feat.reshape((-1,100))
            edges_feature.append(edge_feat)
            qry_node = qrys_node[i]
            doc_node = docs_node[i]
            qry_node = {n:idx for idx,n in enumerate(qry_node)}
            doc_node = {n:idx for idx,n in enumerate(doc_node)}
            qry_mp = qrys_mp[i]
            doc_mp = docs_mp[i]
            qry_span = qrys_span[i]
            doc_span = docs_span[i]
            for idx,n in enumerate(qry_node):
                word_idx = qry_span[idx][0]
                ernie_idx = qry_mp[word_idx]
                node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                nodes[n] = i*self.cfg['max_seq_len']+ernie_idx
            for idx,n in enumerate(doc_node):
                word_idx = doc_span[idx][0]
                ernie_idx = doc_mp.get(word_idx, -1)
                if ernie_idx!=-1:
                    node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                    nodes[n]=i*self.cfg['max_seq_len']+ernie_idx
            for source,target in graph.edges:
                new_source=0  #在大graph中的位置
                new_target=0  #在大graph中的位置
                if nodes.get(source, -1)!=-1:
                    new_source = nodes.get(source)
                else:
                    new_source=node_num
                    node_num+=1
                    concept_feats.append(node_feat[source])
                    nodes[source] = new_source
                ###################下面是target#######################
                if nodes.get(target,-1)!=-1:
                    new_target = nodes.get(target)
                else:
                    new_target=node_num
                    node_num+=1
                    concept_feats.append(node_feat[target])
                    nodes[target]=new_target
                edges.append((new_source,new_target))
        if len(concept_feats):
            concept_feats=np.stack(concept_feats,axis=0)
            nodes_feature=np.concatenate([node_feats,concept_feats],axis=0)
        else:
            nodes_feature = node_feats
        edges_feature=np.concatenate(edges_feature,axis=0)
        graph = pgl.Graph(num_nodes=node_num,
                edges=edges,
                node_feat={"feature": nodes_feature},
                edge_feat={"edge_feature":edges_feature})
        return graph
    def _collate_fn(self, sample_list):
        '''
        qid, pid, label, input_ids, token_type_ids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp
        '''
        tmp = list(zip(*sample_list))
        batch = [np.concatenate(s, axis=0) for s in tmp[:5]]
        graphs = []
        for g in tmp[5]:
            graphs+=g
        qrys_span = []
        for s in tmp[6]:
            qrys_span+=s
        docs_span = []
        for s in tmp[7]:
            docs_span+=s
        qrys_node = []
        for n in tmp[8]:
            qrys_node+=n
        docs_node = []
        for n in tmp[9]:
            docs_node+=n
        qrys_mp = []
        for m in tmp[10]:
            qrys_mp+=(m)
        docs_mp = []
        for m in tmp[11]:
            docs_mp+=(m)
        batched_graph = self.batched_graph(graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span)
        batch.append(batched_graph)
        return batch
        
def apply_mask(input_ids,mask_rate=0.15):
    x,y = np.where(input_ids==102)
    y_sep = y[::2]
    y_end = y[1::2]
    new_x = np.concatenate([[i]*(v-2) for i,v in enumerate(y_end)]).reshape(-1)
    new_y = np.concatenate([list(range(1,y_sep[i]))+list(range(y_sep[i]+1, v)) for i,v in enumerate(y_end)]).reshape(-1)
    mask_pos = random.choices(range(len(new_x)), k=max(1, int(len(new_x)*mask_rate)))
    mask_pos = new_x[mask_pos],new_y[mask_pos]
    mask_label = input_ids[mask_pos]
    rand = np.random.rand(*mask_pos[0].shape)
    choose_original = rand < 0.1  #
    choose_random_id = (0.1 < rand) & (rand < 0.2)  #
    choose_mask_id = 0.2 < rand  #
    random_id = np.random.randint(1, 30522, size=mask_pos[0].shape)
    
    replace_id = 103 * choose_mask_id + \
             random_id * choose_random_id + \
             mask_label * choose_original
    input_ids[mask_pos] = replace_id
    return input_ids, np.stack(mask_pos, -1), mask_label



class PreTrainErnieConceptDataset(IterableDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_procs = dist.get_world_size()  # 8
        self.rank = dist.get_rank()  # rank
        self.datas = [tarfile.open(f) for f in glob.glob(self.cfg['pretrain_input_file'])]
        self.batch_size = cfg['pretrain_batch_size']
    @property
    def get_batch(self):
        batch = []
        for data in self.datas:
            for sample in data:
                if 'sample' in sample.get_info()['name']:
                    sample = pkl.load(data.extractfile(sample))
                    batch.append(sample)
                    if len(batch)==self.batch_size*self.n_procs:
                        yield batch
                        batch = []
        if len(batch) and len(batch)<self.batch_size*self.n_procs:
            batch*=self.batch_size*self.n_procs
            batch = batch[:self.batch_size*self.n_procs]
            yield batch
            batch = []
    def __iter__(self):
        for batch in self.get_batch:
            batch = batch[self.rank::self.n_procs]
            try:
                batch = self._collate_fn(batch)
                yield batch
            except:
                print(batch)
                pkl.dump(batch, open("batch.pkl",'wb'))
    def batched_graph(self, graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span):
        '''
        graphs:包含40*4个graph的list
        return:一个大graph，前160*40*4个node和ernie对应，后面的node是知识图谱中的
        '''
        node_num = len(graphs)*self.cfg['max_seq_len']
        node_feats = np.zeros((node_num,100))
        edges = []
        edges_feature = []
        concept_feats = []
        for i,graph in enumerate(graphs):
            node_feat = graph.node_feat['feature']
            edge_feat = graph.edge_feat['edge_feature']
            nodes = {}
            edge_feat = edge_feat.reshape((-1,100))
            edges_feature.append(edge_feat)
            qry_node = qrys_node[i]
            doc_node = docs_node[i]
            qry_node = {n:idx for idx,n in enumerate(qry_node)}
            doc_node = {n:idx for idx,n in enumerate(doc_node)}
            qry_mp = qrys_mp[i]
            doc_mp = docs_mp[i]
            qry_span = qrys_span[i]
            doc_span = docs_span[i]
            for idx,n in enumerate(qry_node):
                word_idx = qry_span[idx][0]
                ernie_idx = qry_mp.get(word_idx, -1)
                if ernie_idx!=-1:
                    node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                    nodes[n] = i*self.cfg['max_seq_len']+ernie_idx
            for idx,n in enumerate(doc_node):
                word_idx = doc_span[idx][0]
                ernie_idx = doc_mp.get(word_idx, -1)
                if ernie_idx!=-1:
                    node_feats[i*self.cfg['max_seq_len']+ernie_idx]=node_feat[n]
                    nodes[n]=i*self.cfg['max_seq_len']+ernie_idx
            for source,target in graph.edges:
                new_source=0  #在大graph中的位置
                new_target=0  #在大graph中的位置
                if nodes.get(source, -1)!=-1:
                    new_source = nodes.get(source)
                else:
                    new_source=node_num
                    node_num+=1
                    concept_feats.append(node_feat[source])
                    nodes[source] = new_source
                ###################下面是target#######################
                if nodes.get(target,-1)!=-1:
                    new_target = nodes.get(target)
                else:
                    new_target=node_num
                    node_num+=1
                    concept_feats.append(node_feat[target])
                    nodes[target]=new_target
                edges.append((new_source,new_target))
        if len(concept_feats):
            concept_feats=np.stack(concept_feats,axis=0)
            nodes_feature=np.concatenate([node_feats,concept_feats],axis=0)
        else:
            nodes_feature = node_feats
        edges_feature=np.concatenate(edges_feature,axis=0)
        graph = pgl.Graph(num_nodes=node_num,
                edges=edges,
                node_feat={"feature": nodes_feature},
                edge_feat={"edge_feature":edges_feature})
        return graph
    
    
    def _collate_fn(self, sample_list):
        '''
        input_ids, token_type_ids, label, qids, graph, q_span, d_span, qry_node,doc_node, qry_mp, doc_mp
        '''
        tmp = list(zip(*sample_list))
        tmp[0] = list(tmp[0])
        tmp[1:3] = [np.stack(tmp[i],axis=0) for i in range(1,3)]
        labels,input_ids,seg_ids,graphs,qrys_span,docs_span,qrys_node,docs_node,qrys_mp,docs_mp = tmp
        input_ids, mask_pos, mask_label = apply_mask(input_ids)
        batch = [np.array(labels), input_ids, seg_ids, mask_pos, mask_label]
        batched_graph = self.batched_graph(graphs,qrys_node,docs_node, qrys_mp, docs_mp, qrys_span, docs_span)
        batch.append(batched_graph)
        return batch