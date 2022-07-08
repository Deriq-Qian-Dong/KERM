#!/bin/bash
model_name=$1
queue=$2
model_size=$3
sample_num=$4
if [[ ${queue} =~ "v100" ]]; then
    batch_size=`expr 40 / $sample_num`
fi 
if [[ ${queue} =~ "a100" ]]; then
    batch_size=`expr 80 / $sample_num`
fi 
if [[ ${model_size} =~ "base" ]]; then
    batch_size=`expr $batch_size \* 2`
fi 
echo "batch size ${batch_size}"
dev_batch_size=1
pretrain_input_file='data/pretrain/*'
train_input_file='data/train.concept.v2.tar.gz'  #训练数据
dev_input_file='data/dev.concept.v2.tar.gz'  #测试数据
warmup_proportion=0.2
eval_step_proportion=0.01
report_step=10
epoch=5
# eval_step_proportion=`echo "scale=5; 1/$epoch" | bc`
### 下面是永远不用改的
min_index=25
max_index=768
max_seq_len=160
collection='data/collection.tsv'  
ernie_config_file=${model_size}/ernie_config.json  #ernie配置文件
vocab_file=${model_size}/vocab.txt  #ernie配置文件
# warm_start_from=data/reranker-4gpu-5-2.p  #ernie参数
warm_start_from=data/erniebest.p
qrels='data/qrels.train.tsv'
query='data/train.query.txt'
resource='data/concept.txt'
cpnet='data/conceptnet.en.pruned.graph'
pattern_path='data/matcher_patterns.json'
word2vec='data/GoogleNews-vectors-negative300.bin.gz'
ent_emb='data/glove.transe.sgd.ent.npy'
rel_emb='data/glove.transe.sgd.rel.npy'
books='data/books.txt'
cnts='data/cnts.pkl'
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
pip install networkx
pip install pgl
pip install spacy
pip install nltk
pip install gensim
python -m spacy download en_core_web_sm
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
for ((i=0;i<26;i++))
do
    nohup python ernie/count.py --ernie_config_file=${ernie_config_file} --books=${books} --cnts=${cnts} --local_rank=${i} > output/log/log.${i} 2>&1 &
done

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="
# upload
echo "Starting uploading file to HDFS"
# tar -zcvf /root/paddlejob/workspace/env_run/output.tar.gz gen_data/
${hdfs_cmd} -mkdir /user/sasd-adv/diaoyan/user/modelzoo/${model_name}
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/output /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/ernie /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/script /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
echo "Done uploading file to HDFS"
