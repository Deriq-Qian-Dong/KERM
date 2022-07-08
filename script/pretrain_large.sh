#!/bin/bash
model_name=$1
queue=$2
model_size=$3
dataset=marco
sample_num=20
if [[ ${dataset} =~ "treccar" ]]; then
    sample_num=10
fi 
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
batch_size=8
pretrain_batch_size=64
run_func=pretrain
dev_batch_size=1
pretrain_input_file='data/pretrain80/*'
train_input_file=data/${dataset}/train.concept.tar.gz  #训练数据
dev_input_file=data/${dataset}/dev.concept.dl2019.tar.gz  #测试数据
instance_num=502939  #v3: 917012
sample_range=20
if [[ ${dataset} =~ "treccar" ]]; then
    instance_num=2806552
    sample_range=10
fi 
warmup_proportion=0.2
eval_step_proportion=0.01
report_step=10
epoch=5
# eval_step_proportion=`echo "scale=5; 1/$epoch" | bc`
### 下面是永远不用改的
min_index=25
max_index=768
max_seq_len=80
collection=data/${dataset}/collection.tsv
ernie_config_file=${model_size}/ernie_config.json  #ernie配置文件
vocab_file=${model_size}/vocab.txt  #ernie配置文件
# warm_start_from=data/reranker-4gpu-5-2.p  #ernie参数
# warm_start_from=data/kgbest.p
# warm_start_from=data/${dataset}/reranker-4gpu-5.p
warm_start_from=data/${dataset}/ernie_large.p
qrels=data/${dataset}/qrels.tsv
query=data/${dataset}/train.query.txt
resource='data/concept.txt'
cpnet='data/conceptnet.en.pruned.graph'
pattern_path='data/matcher_patterns.json'
word2vec='data/GoogleNews-vectors-negative300.bin.gz'
ent_emb='data/glove.transe.sgd.ent.npy'
rel_emb='data/glove.transe.sgd.rel.npy'
books='data/books.txt'
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
rm -rf /etc/pip.conf
cp pip.conf /etc/pip.conf
pip install networkx
pip install pgl
pip install spacy
pip install nltk
pip install gensim
pip install data/gensim-4.1.2-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
python -m spacy download en_core_web_sm
pip install data/en_core_web_sm-3.2.0-py3-none-any.whl
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
python -m paddle.distributed.launch \
    --log_dir ${log_dir} \
    ernie/finetune.py \
    --train_input_file=${train_input_file} \
    --ernie_config_file=${ernie_config_file} \
    --vocab_file=${vocab_file} \
    --resource=${resource} \
    --cpnet=${cpnet} \
    --dev_input_file=${dev_input_file} \
    --warm_start_from=${warm_start_from} \
    --batch_size=${batch_size} \
    --warmup_proportion=${warmup_proportion} \
    --eval_step_proportion=${eval_step_proportion} \
    --report=${report_step} \
    --qrels=${qrels} \
    --query=${query} \
    --collection=${collection} \
    --top1000=${top1000} \
    --min_index=${min_index} \
    --max_index=${max_index} \
    --epoch=${epoch} \
    --sample_num=${sample_num} \
    --dev_batch_size=${dev_batch_size} \
    --pretrain_input_file=${pretrain_input_file} \
    --ent_emb=${ent_emb} \
    --rel_emb=${rel_emb} \
    --pattern_path=${pattern_path} \
    --word2vec=${word2vec} \
    --instance_num=${instance_num} \
    --dataset=${dataset} \
    --sample_range=${sample_range} \
    --max_seq_len=${max_seq_len} \
    --run_func=${run_func} \
    --pretrain_batch_size=${pretrain_batch_size} \
    --num_gnn_layers=3

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="
# upload
echo "Starting uploading file to HDFS"
# tar -zcvf /root/paddlejob/workspace/env_run/output.tar.gz gen_data/
${hdfs_cmd} -mkdir /user/sasd-adv/diaoyan/user/modelzoo/${model_name}
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/output /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/ernie /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/script /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
echo "Done uploading file to HDFS"
