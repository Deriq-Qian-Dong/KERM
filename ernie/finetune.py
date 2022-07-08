from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '1.0'
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
from ernie_concept import ErnieWithConcept,pretrainedErnieWithConcept,ErnieWithConceptv2
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

def define_args():
    parser = argparse.ArgumentParser('kg-ERNIE-rerank model')
    parser.add_argument('--run', type=str, default="nce")
    parser.add_argument('--model', type=str, default="ErnieWithGNN")
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    parser.add_argument('--ernie_config_file', type=str, default="/home/user/codes/kgreranker/base/ernie_config.json")
    parser.add_argument('--vocab_file', type=str, default="/home/user/codes/kgreranker/base/vocab.txt")
    parser.add_argument('--train_input_file', type=str,default="/home/user/hdfs_data/data_train/train.concept.tar.gz")
    parser.add_argument('--dev_input_file', type=str, default="/home/user/hdfs_data/data_train/dev.concept.tar.gz")
    parser.add_argument('--test_input_file', type=str, default="")
    parser.add_argument('--pretrain_input_file', type=str, default="/home/user/hdfs_data/data_train/pretrain/*")
    parser.add_argument('--pretrain_batch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dev_batch_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=160)
    parser.add_argument('--warm_start_from', type=str, default="/home/user/hdfs_data/data_train/ernie_large.p")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--eval_step_proportion', type=float, default=1.0)
    parser.add_argument('--ernie_lr', type=float, default=1e-5)
    parser.add_argument('--report', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--qrels', type=str, default="/home/user/hdfs_data/data_train/qrels.train.tsv")
    parser.add_argument('--top1000', type=str, default="/home/user/hdfs_data/data_train/train.concept.gz")
    parser.add_argument('--collection', type=str, default="/home/user/hdfs_data/data_train/collection.tsv")
    parser.add_argument('--query', type=str, default="/home/user/hdfs_data/data_train/train.query.txt")
    parser.add_argument('--min_index', type=int, default=25)
    parser.add_argument('--max_index', type=int, default=768)
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--kerm_version', type=str, default="v1")
    parser.add_argument('--run_func', type=str, default="finetune")


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
    parser.add_argument('--word2vec', type=str, default="/home/user/hdfs_data/data_train/GoogleNews-vectors-negative300.bin.gz")
    parser.add_argument('--topk_sents', type=int, default=1)
    parser.add_argument('--ent_emb', type=str, default="/home/user/hdfs_data/concept_net/glove.transe.sgd.ent.npy")
    parser.add_argument('--dataset', type=str, default="marco")
    parser.add_argument('--rel_emb', type=str, default="/home/user/hdfs_data/concept_net/glove.transe.sgd.rel.npy")
    parser.add_argument('--gnn_hidden_size', type=int, default=100)
    parser.add_argument('--instance_num', type=int, default=109)# 502939
    parser.add_argument('--sample_range', type=int, default=20)
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    return args

def run_concept(args, all_configs, split_concept=False):
    cfg = all_configs
    local_rank = paddle.distributed.get_rank()
    _nranks = ParallelEnv().nranks  
    args.learning_rate = args.learning_rate * (_nranks/8)
    all_configs['learning_rate'] = args.learning_rate
    all_configs['ernie_lr'] = all_configs['ernie_lr']/all_configs['learning_rate']
    # dataset
    print(args.batch_size)
    train_dataset = dataset_factory.TrainErnieConceptDataset(all_configs)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=None, num_workers=0, return_list=True)
    dev_dataset = dataset_factory.DevErnieConceptDataset(all_configs)
    dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=1, collate_fn=dev_dataset._collate_fn,num_workers=0, drop_last=True)
    if all_configs['test_input_file']:
        test_dataset = dataset_factory.DevErnieConceptDataset(all_configs, is_test=True)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset._collate_fn,num_workers=0, drop_last=True)
    
    # model
    if split_concept:
        model = ErnieWithConceptv2(all_configs)
    else:
        model = ErnieWithConcept(all_configs)
    state_dict=paddle.load(all_configs['warm_start_from'])
    if all_configs['kerm_version']=='v2':
        new_state_dict = {}
        for key in state_dict.keys():
            if 'ffn.gnn' in key:
                new_state_dict[key.replace('ffn.','')] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        model.load_dict(new_state_dict)
    else:
        model.load_dict(state_dict)
    all_steps_per_epoch = int(args.instance_num/args.batch_size/_nranks)+_nranks
    eval_step = all_steps_per_epoch//5
    all_steps = all_steps_per_epoch*all_configs['epoch']
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "ln",'norms'])]
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=all_configs['learning_rate'], warmup_steps=int(all_configs['warmup_proportion']*all_steps), start_lr=all_configs['learning_rate']/1000, end_lr=all_configs['learning_rate'], verbose=False)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=all_configs['weight_decay'],apply_decay_param_fun=lambda x:x in decay_params)
    ce_loss = paddle.nn.CrossEntropyLoss(reduction='mean')

    dist.init_parallel_env()
    model = paddle.DataParallel(model)
    total_loss = 0
    best_mrr = 0
    os.makedirs('output',exist_ok=True)
    metrics_mrr = Mrr(cfg)
    test_mrr = Mrr(cfg)
    train_metrics_mrr = Mrr(cfg)
    scaler = paddle.amp.GradScaler(init_loss_scaling=128)
    start = time.time()
    local_start = time.time()
    skip_eval_step = 0
    eval_cnts = 0
    for epoch in range(all_configs['epoch']):
        train_metrics_mrr.reset()
        total_loss = 0
        if epoch==1:
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=all_configs['learning_rate'],eta_min=all_configs['learning_rate']/10000, T_max=all_steps, verbose=False)
            optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=all_configs['weight_decay'],apply_decay_param_fun=lambda x:x in decay_params)
        for step,td in enumerate(train_loader):
            # if ('large' in all_configs['warm_start_from'] and _nranks==4 and 12070<=step<=12080) or ('large' in all_configs['warm_start_from'] and _nranks==8 and 15710<step):  #OOM for large
            #     continue
            (pair_input_ids, pair_token_type_ids, labels, qid, g) = td
            if all_configs['fp16']:
                with paddle.amp.auto_cast():
                    output = model(pair_input_ids, pair_token_type_ids, g)
                    output = output.reshape((-1, all_configs['sample_num']))
                    labels = paddle.to_tensor(labels)
                    labels.stop_gradient = True
                    loss = ce_loss(output, labels)
            else:
                output = model(pair_input_ids, pair_token_type_ids, g)
                output = output.reshape((-1, all_configs['sample_num']))
                labels = paddle.to_tensor(labels)
                labels.stop_gradient = True
                loss = ce_loss(output, labels)

            if _nranks>1:
                loss = _all_gather(loss, _nranks)
                output = _all_gather(output, _nranks)
                qid = _all_gather(qid, _nranks)
            labels = np.array([[1]+[0]*(all_configs['sample_num']-1) for _ in range(qid.shape[0])])
            train_metrics_mrr.update(qid,qid,labels,output)
            if all_configs['fp16']:
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
            else:
                loss.backward()
                optimizer.step()
            optimizer.clear_grad()
            scheduler.step()
            step+=1
            total_loss += loss.numpy()[0]
            if step%args.report==0 and local_rank==0:
                seconds = time.time()-local_start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                local_start = time.time()
                if all_configs['dataset']=="marco":
                    train_mrr = train_metrics_mrr.accumulate()
                else:
                    train_mrr = train_metrics_mrr.accumulate_map()
                print("epoch:%d training step: %d/%d, mean loss: %.5f, current loss: %.5f, mrr: %.5f, lr: %.10f,"%(epoch, step, all_steps_per_epoch, total_loss/step, loss.numpy()[0], train_mrr, optimizer.get_lr()),"report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
                seconds = time.time()-start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
                print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
            if step%eval_step==0:
                eval_cnts+=1
                if eval_cnts<=skip_eval_step:
                    continue
                with paddle.no_grad():
                    metrics_mrr.reset()
                    for step_eval,data in enumerate(dev_loader):
                        # print(step_eval,time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
                        if all_configs['fp16']:
                            with paddle.amp.auto_cast():
                                qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                                score = model(pair_input_ids, pair_token_type_ids, g)
                        else:
                            qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                            score = model(pair_input_ids, pair_token_type_ids, g)
                        if _nranks>1:
                            if qid.shape[0]!=64:
                                qid=paddle.concat([qid, -100000000*paddle.ones((64-qid.shape[0],)).astype("int")])
                                pid=paddle.concat([pid, -100000000*paddle.ones((64-pid.shape[0],)).astype("int")])
                                label=paddle.concat([label, -100000000*paddle.ones((64-label.shape[0],)).astype("int")])
                                score=paddle.concat([score, -100000000*paddle.ones((64-score.shape[0],1)).astype("float32")],axis=0)
                            qid = _all_gather(qid, _nranks)
                            pid = _all_gather(pid, _nranks)
                            label = _all_gather(label, _nranks)
                            score = _all_gather(score, _nranks)
                        score = score.numpy()
                        metrics_mrr.update(qid, pid, label, score)
                    
                    mrr = metrics_mrr.accumulate()
                    MAP = metrics_mrr.accumulate_map()
                    NDCG = metrics_mrr.accumulate_ndcg()

                    print(metrics_mrr.qid_saver.shape)
                    print(metrics_mrr.label_saver.shape)
                    print(metrics_mrr.pred_saver.shape)
                    if all_configs['test_input_file']:
                        test_mrr.reset()
                        for step_eval,data in enumerate(test_loader):
                            # print(step_eval,time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
                            if all_configs['fp16']:
                                with paddle.amp.auto_cast():
                                    qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                                    score = model(pair_input_ids, pair_token_type_ids, g)
                            else:
                                qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                                score = model(pair_input_ids, pair_token_type_ids, g)
                            if _nranks>1:
                                if qid.shape[0]!=64:
                                    qid=paddle.concat([qid, -100000000*paddle.ones((64-qid.shape[0],)).astype("int")])
                                    pid=paddle.concat([pid, -100000000*paddle.ones((64-pid.shape[0],)).astype("int")])
                                    label=paddle.concat([label, -100000000*paddle.ones((64-label.shape[0],)).astype("int")])
                                    score=paddle.concat([score, -100000000*paddle.ones((64-score.shape[0],1)).astype("float32")],axis=0)
                                qid = _all_gather(qid, _nranks)
                                pid = _all_gather(pid, _nranks)
                                label = _all_gather(label, _nranks)
                                score = _all_gather(score, _nranks)
                            score = score.numpy()
                            test_mrr.update(qid, pid, label, score)
                    if local_rank==0:
                        pkl.dump(metrics_mrr, open("output/dev_mrr_%.4f_%d.pkl"%(mrr,eval_cnts), "wb"))
                        pkl.dump(test_mrr, open("output/test_mrr_%.4f_%d.pkl"%(mrr,eval_cnts), "wb"))
                    if mrr>best_mrr:
                        print("*"*50)
                        print("new top")
                        print("*"*50)
                        best_mrr = mrr
                        paddle.save(model.state_dict(), "output/reranker.p")
                        pkl.dump(metrics_mrr, open("output/mrr.pkl", "wb"))
                        pkl.dump(test_mrr, open("output/test_mrr.pkl", "wb"))
                    
                    seconds = time.time()-local_start
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    local_start = time.time()
                    print("******************eval, mrr@10: %.10f,"%(mrr),"report used time:%02d:%02d:%02d," % (h, m, s), end=" ")
                    print("******************eval, MAP: %.10f,"%(MAP),"report used time:%02d:%02d:%02d," % (h, m, s), end=" ")
                    print("******************eval, NDCG@10: %.10f,"%(NDCG),"report used time:%02d:%02d:%02d," % (h, m, s), end=" ")
                    seconds = time.time()-start
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
                    print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))


def run_concept_with_mlm(args, all_configs, split_concept=False):
    cfg = all_configs
    local_rank = paddle.distributed.get_rank()
    _nranks = ParallelEnv().nranks  
    args.learning_rate = args.learning_rate * (_nranks/8)
    all_configs['learning_rate'] = args.learning_rate
    all_configs['ernie_lr'] = all_configs['ernie_lr']/all_configs['learning_rate']
    # dataset
    print(args.batch_size)
    train_dataset = dataset_factory.TrainErnieConceptWithMLMDataset(all_configs)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=None, num_workers=0, return_list=True)
    dev_dataset = dataset_factory.DevErnieConceptDataset(all_configs)
    dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=1, collate_fn=dev_dataset._collate_fn,num_workers=0, drop_last=True)
    if all_configs['test_input_file']:
        test_dataset = dataset_factory.DevErnieConceptDataset(all_configs, is_test=True)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset._collate_fn,num_workers=0, drop_last=True)
    
    # model
    if split_concept:
        model = ErnieWithConceptv2(all_configs)
    else:
        model = ErnieWithConcept(all_configs)
    state_dict=paddle.load(all_configs['warm_start_from'])
    model.load_dict(state_dict)
    all_steps_per_epoch = int(args.instance_num/args.batch_size/_nranks)+_nranks
    eval_step = all_steps_per_epoch//5
    all_steps = all_steps_per_epoch*all_configs['epoch']
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "ln",'norms'])]
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=all_configs['learning_rate'], warmup_steps=int(all_configs['warmup_proportion']*all_steps), start_lr=all_configs['learning_rate']/1000, end_lr=all_configs['learning_rate'], verbose=False)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=all_configs['weight_decay'],apply_decay_param_fun=lambda x:x in decay_params)
    ce_loss = paddle.nn.CrossEntropyLoss(reduction='mean')

    dist.init_parallel_env()
    model = paddle.DataParallel(model)
    total_loss = 0
    best_mrr = 0
    os.makedirs('output',exist_ok=True)
    metrics_mrr = Mrr(cfg)
    test_mrr = Mrr(cfg)
    train_metrics_mrr = Mrr(cfg)
    scaler = paddle.amp.GradScaler(init_loss_scaling=128)
    start = time.time()
    local_start = time.time()
    skip_eval_step = 7
    eval_cnts = 0
    mlm_ratio = 0.2
    for epoch in range(all_configs['epoch']):
        train_metrics_mrr.reset()
        total_loss = 0
        if epoch==1:
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=all_configs['learning_rate'],eta_min=all_configs['learning_rate']/10000, T_max=all_steps, verbose=False)
            optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=all_configs['weight_decay'],apply_decay_param_fun=lambda x:x in decay_params)
        for step,td in enumerate(train_loader):
            (pair_input_ids, pair_token_type_ids, labels, qid, g, mask_pos, mask_label) = td
            if all_configs['fp16']:
                with paddle.amp.auto_cast():
                    output,logits_2d = model(pair_input_ids, pair_token_type_ids, g, with_mlm=True, mlm_pos=mask_pos)
                    output = output.reshape((-1, all_configs['sample_num']))
                    labels = paddle.to_tensor(labels)
                    labels.stop_gradient = True
                    loss = ce_loss(output, labels)
                    mlm_loss = ce_loss(logits_2d, mask_label)
                    loss = loss + mlm_ratio*mlm_loss
            else:
                output,logits_2d = model(pair_input_ids, pair_token_type_ids, g, with_mlm=True, mlm_pos=mask_pos)
                output = output.reshape((-1, all_configs['sample_num']))
                labels = paddle.to_tensor(labels)
                labels.stop_gradient = True
                loss = ce_loss(output, labels)
                mlm_loss = ce_loss(logits_2d, mask_label)
                loss = loss + mlm_ratio*mlm_loss

            if _nranks>1:
                loss = _all_gather(loss, _nranks)
                output = _all_gather(output, _nranks)
                qid = _all_gather(qid, _nranks)
            labels = np.array([[1]+[0]*(all_configs['sample_num']-1) for _ in range(qid.shape[0])])
            train_metrics_mrr.update(qid,qid,labels,output)
            if all_configs['fp16']:
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
            else:
                loss.backward()
                optimizer.step()
            optimizer.clear_grad()
            scheduler.step()
            step+=1
            total_loss += loss.numpy()[0]
            if step%args.report==0 and local_rank==0:
                seconds = time.time()-local_start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                local_start = time.time()
                if all_configs['dataset']=="marco":
                    train_mrr = train_metrics_mrr.accumulate()
                else:
                    train_mrr = train_metrics_mrr.accumulate_map()
                print("epoch:%d training step: %d/%d, mean loss: %.5f, current loss: %.5f, mrr: %.5f, lr: %.10f,"%(epoch, step, all_steps_per_epoch, total_loss/step, loss.numpy()[0], train_mrr, optimizer.get_lr()),"report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
                seconds = time.time()-start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
                print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
            if step%eval_step==0:
                eval_cnts+=1
                if eval_cnts<=skip_eval_step:
                    continue
                with paddle.no_grad():
                    metrics_mrr.reset()
                    for step_eval,data in enumerate(dev_loader):
                        # print(step_eval,time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
                        if all_configs['fp16']:
                            with paddle.amp.auto_cast():
                                qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                                score = model(pair_input_ids, pair_token_type_ids, g)
                        else:
                            qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                            score = model(pair_input_ids, pair_token_type_ids, g)
                        if _nranks>1:
                            if qid.shape[0]!=64:
                                qid=paddle.concat([qid, -100000000*paddle.ones((64-qid.shape[0],)).astype("int")])
                                pid=paddle.concat([pid, -100000000*paddle.ones((64-pid.shape[0],)).astype("int")])
                                label=paddle.concat([label, -100000000*paddle.ones((64-label.shape[0],)).astype("int")])
                                score=paddle.concat([score, -100000000*paddle.ones((64-score.shape[0],1)).astype("float32")],axis=0)
                            qid = _all_gather(qid, _nranks)
                            pid = _all_gather(pid, _nranks)
                            label = _all_gather(label, _nranks)
                            score = _all_gather(score, _nranks)
                        score = score.numpy()
                        metrics_mrr.update(qid, pid, label, score)
                    
                    mrr = metrics_mrr.accumulate()
                    MAP = metrics_mrr.accumulate_map()
                    NDCG = metrics_mrr.accumulate_ndcg()

                    print(metrics_mrr.qid_saver.shape)
                    print(metrics_mrr.label_saver.shape)
                    print(metrics_mrr.pred_saver.shape)
                    if mrr>best_mrr:
                        print("*"*50)
                        print("new top")
                        print("*"*50)
                        best_mrr = mrr
                        if all_configs['test_input_file']:
                            test_mrr.reset()
                            for step_eval,data in enumerate(test_loader):
                                # print(step_eval,time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
                                if all_configs['fp16']:
                                    with paddle.amp.auto_cast():
                                        qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                                        score = model(pair_input_ids, pair_token_type_ids, g)
                                else:
                                    qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                                    score = model(pair_input_ids, pair_token_type_ids, g)
                                if _nranks>1:
                                    if qid.shape[0]!=64:
                                        qid=paddle.concat([qid, -100000000*paddle.ones((64-qid.shape[0],)).astype("int")])
                                        pid=paddle.concat([pid, -100000000*paddle.ones((64-pid.shape[0],)).astype("int")])
                                        label=paddle.concat([label, -100000000*paddle.ones((64-label.shape[0],)).astype("int")])
                                        score=paddle.concat([score, -100000000*paddle.ones((64-score.shape[0],1)).astype("float32")],axis=0)
                                    qid = _all_gather(qid, _nranks)
                                    pid = _all_gather(pid, _nranks)
                                    label = _all_gather(label, _nranks)
                                    score = _all_gather(score, _nranks)
                                score = score.numpy()
                                test_mrr.update(qid, pid, label, score)
                        if local_rank==0:
                            paddle.save(model.state_dict(), "output/reranker.p")
                            pkl.dump(metrics_mrr, open("output/mrr.pkl", "wb"))
                            pkl.dump(test_mrr, open("output/test_mrr.pkl", "wb"))
                    
                    seconds = time.time()-local_start
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    local_start = time.time()
                    print("******************eval, mrr@10: %.10f,"%(mrr),"report used time:%02d:%02d:%02d," % (h, m, s), end=" ")
                    print("******************eval, MAP: %.10f,"%(MAP),"report used time:%02d:%02d:%02d," % (h, m, s), end=" ")
                    print("******************eval, NDCG@10: %.10f,"%(NDCG),"report used time:%02d:%02d:%02d," % (h, m, s), end=" ")
                    seconds = time.time()-start
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
                    print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))



def eval_concept(args, all_configs):
    cfg = all_configs
    local_rank = paddle.distributed.get_rank()
    _nranks = ParallelEnv().nranks  
    args.learning_rate = args.learning_rate * (_nranks/8)
    all_configs['learning_rate'] = args.learning_rate
    all_configs['ernie_lr'] = all_configs['ernie_lr']/all_configs['learning_rate']
    # dataset
    print(args.batch_size)
    train_dataset = dataset_factory.TrainErnieConceptDataset(all_configs)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=None, num_workers=0, return_list=True)
    dev_dataset = dataset_factory.DevErnieConceptDataset(all_configs)
    dev_loader = paddle.io.DataLoader(dev_dataset, batch_size=1, collate_fn=dev_dataset._collate_fn,num_workers=0, drop_last=True)
    
    # model
    model = ErnieWithConcept(all_configs)
    state_dict=paddle.load(all_configs['warm_start_from'])
    model.load_dict(state_dict)
    all_steps_per_epoch = int(args.instance_num/args.batch_size/_nranks)+_nranks
    all_steps = all_steps_per_epoch*all_configs['epoch']
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "ln",'norms'])]
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=all_configs['learning_rate'], warmup_steps=int(all_configs['warmup_proportion']*all_steps), start_lr=all_configs['learning_rate']/1000, end_lr=all_configs['learning_rate'], verbose=False)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=all_configs['weight_decay'],apply_decay_param_fun=lambda x:x in decay_params)
    ce_loss = paddle.nn.CrossEntropyLoss(reduction='mean')

    dist.init_parallel_env()
    model = paddle.DataParallel(model)
    total_loss = 0
    best_mrr = 0
    os.makedirs('output',exist_ok=True)
    metrics_mrr = Mrr(cfg)
    train_metrics_mrr = Mrr(cfg)
    scaler = paddle.amp.GradScaler(init_loss_scaling=128)
    start = time.time()
    local_start = time.time()
    
    with paddle.no_grad():
        metrics_mrr.reset()
        for step_eval,data in enumerate(dev_loader):
            # print(step_eval,time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
            if all_configs['fp16']:
                with paddle.amp.auto_cast():
                    qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                    score = model(pair_input_ids, pair_token_type_ids, g)
            else:
                qid, pid, label, pair_input_ids, pair_token_type_ids, g = data
                score = model(pair_input_ids, pair_token_type_ids, g)
            if _nranks>1:
                if qid.shape[0]!=64:
                    qid=paddle.concat([qid, -100000000*paddle.ones((64-qid.shape[0],)).astype("int")])
                    pid=paddle.concat([pid, -100000000*paddle.ones((64-pid.shape[0],)).astype("int")])
                    label=paddle.concat([label, -100000000*paddle.ones((64-label.shape[0],)).astype("int")])
                    score=paddle.concat([score, -100000000*paddle.ones((64-score.shape[0],1)).astype("float32")],axis=0)
                qid = _all_gather(qid, _nranks)
                pid = _all_gather(pid, _nranks)
                label = _all_gather(label, _nranks)
                score = _all_gather(score, _nranks)
            score = score.numpy()
            metrics_mrr.update(qid, pid, label, score)
        if all_configs['dataset']=="marco":
            mrr = metrics_mrr.accumulate()
        else:
            mrr = metrics_mrr.accumulate_map()
        print(metrics_mrr.qid_saver.shape)
        print(metrics_mrr.label_saver.shape)
        print(metrics_mrr.pred_saver.shape)
        if mrr>best_mrr:
            print("*"*50)
            print("new top")
            print("*"*50)
            best_mrr = mrr
            if local_rank==0:
                paddle.save(model.state_dict(), "output/reranker.p")
                pkl.dump(metrics_mrr, open("output/mrr.pkl", "wb"))
        seconds = time.time()-local_start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        local_start = time.time()
        print("******************eval, mrr@10: %.10f,"%(mrr),"report used time:%02d:%02d:%02d," % (h, m, s), end=" ")
        seconds = time.time()-start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
        print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))


def run_pretrain(args, all_configs):
    local_rank = paddle.distributed.get_rank()
    _nranks = ParallelEnv().nranks  
    args.learning_rate = args.learning_rate * (_nranks/8)
    all_configs['learning_rate'] = args.learning_rate
    all_configs['ernie_lr'] = all_configs['ernie_lr']/all_configs['learning_rate']
    # dataset
    print(args.batch_size)
    train_dataset = dataset_factory.PreTrainErnieConceptDataset(all_configs)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=None, num_workers=0, return_list=True)  
    # model
    model = pretrainedErnieWithConcept(all_configs)
    state_dict=paddle.load(all_configs['warm_start_from'])
    model.load_dict(state_dict)
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "ln",'norms'])]
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=all_configs['learning_rate'], warmup_steps=1000, start_lr=all_configs['learning_rate']/1000, end_lr=all_configs['learning_rate'], verbose=False)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=all_configs['weight_decay'],apply_decay_param_fun=lambda x:x in decay_params)

    dist.init_parallel_env()
    model = paddle.DataParallel(model)
    total_loss = 0
    os.makedirs('output',exist_ok=True)
    start = time.time()
    local_start = time.time()
    cnt = 0
    for epoch in range(all_configs['epoch']):
        total_loss = 0
        for step,td in enumerate(train_loader):
            (srp_labels, pair_input_ids, pair_token_type_ids, mask_pos, mask_label, g) = td
            loss = model(srp_labels, mask_pos, mask_label, pair_input_ids, pair_token_type_ids, g)

            if _nranks>1:
                loss = _all_gather(loss, _nranks)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            scheduler.step()
            step+=1
            total_loss += loss.numpy()[0]
            if step%args.report==0 and local_rank==0:
                seconds = time.time()-local_start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                local_start = time.time()
                print("epoch:%d training step: %d, mean loss: %.5f, current loss: %.5f, lr: %.10f,"%(epoch, step, total_loss/step, loss.numpy()[0], optimizer.get_lr()),"report used time:%02d:%02d:%02d," % (h, m, s), end=' ')
                seconds = time.time()-start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
                print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
            if step%10000==0 and local_rank==0:
                paddle.save(model.state_dict(), "output/reranker-%d.p"%(cnt))
                cnt+=1

if __name__=="__main__":
    args = define_args()
    all_configs = utils.parse_file(args.ernie_config_file)
    all_configs.update(vars(args))
    all_configs = utils.HParams(**all_configs)
    all_configs.print_config()
    Run_Func_MP = {'pretrain':run_pretrain, 'finetune':run_concept}
    Run_Func_MP[all_configs['run_func']](args, all_configs)
