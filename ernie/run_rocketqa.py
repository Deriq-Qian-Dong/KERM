from paddle.fluid.dygraph.parallel import ParallelEnv
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import rocketqa 
import paddle
import os

def main():
    batch_size = 1024
    local_rank = paddle.distributed.get_rank()
    model = rocketqa.load_model("v2_marco_ce", use_cuda=True, device_id=0, batch_size=batch_size)
    _nranks = ParallelEnv().nranks  
    query = pd.read_csv('/home/user/hdfs_data/data_train/dev.query.txt',sep="\t",header=None)
    query.columns = ['qid','text']
    query.index = query.qid
    query.pop('qid')
    collection = pd.read_csv("/home/user/hdfs_data/data_train/marco/collection.tsv",header=None,sep='\t')
    top1000 = pd.read_csv("/home/user/hdfs_data/data_train/run.bm25.dev.small.tsv",sep="\t",header=None)
    # query = pd.read_csv('data/dev.query.txt',sep="\t",header=None)
    # collection = pd.read_csv("data/marco/collection.tsv",header=None,sep='\t')
    # top1000 = pd.read_csv("data/run.bm25.dev.small.tsv",sep="\t",header=None)
    new_batch_size = batch_size*_nranks

    qrys = []
    psgs = []
    qids = []
    pids = []
    preds = []
    qids_save = []
    pids_save = []
    for i in tqdm(range(len(top1000))):
        qid = top1000.iloc[i][0]
        qids.append(qid)
        pid = top1000.iloc[i][1]
        pids.append(pid)
        qrys.append(query.loc[qid].text)
        psgs.append(collection.iloc[pid][1])
        if (i+1)%new_batch_size==0:
            qrys = qrys[local_rank::_nranks]
            psgs = psgs[local_rank::_nranks]
            qids = qids[local_rank::_nranks]
            pids = pids[local_rank::_nranks]
            scores = model.matching(query=qrys, para=psgs)
            preds+=scores
            qids_save+=qids
            pids_save+=pids
            qrys = []
            psgs = []
            qids = []
            pids = []
    if local_rank==0 and len(qids)!=0:
        scores = model.matching(query=qrys, para=psgs)
        preds+=scores
        qids_save+=qids
        pids_save+=pids
    pkl.dump(qids_save, open("output/qids.%d.pkl"%local_rank,"wb"))
    pkl.dump(pids_save, open("output/pids.%d.pkl"%local_rank,"wb"))
    pkl.dump(preds, open("output/preds.%d.pkl"%local_rank,"wb"))

if __name__=="__main__":
    main()