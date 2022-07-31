#!/bin/bash
model_name=KERMLarge
model_size=large
dataset=marco
sample_num=20
batch_size=`expr 80 / $sample_num`
if [[ ${model_size} =~ "base" ]]; then
    batch_size=`expr $batch_size \* 2`
fi 
echo "batch size ${batch_size}"
dev_batch_size=1
train_input_file=data/${dataset}/train.concept.tar.gz  #训练数据
dev_input_file=data/${dataset}/dev.concept.tar.gz  #测试数据
# test_input_file=data/${dataset}/test.concept.tar.gz
instance_num=502939  #v3: 917012
sample_range=20
warmup_proportion=0.2
eval_step_proportion=0.01
report_step=10
epoch=5
# eval_step_proportion=`echo "scale=5; 1/$epoch" | bc`
### 下面是永远不用改的
min_index=25
max_index=768
max_seq_len=160
collection=data/${dataset}/collection.tsv
ernie_config_file=${model_size}/ernie_config.json  #ernie配置文件
vocab_file=${model_size}/vocab.txt  #ernie配置文件
warm_start_from=data/${dataset}/pretrain/reranker-4gpu-5.p
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
    --test_input_file=${test_input_file} \
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
    --num_gnn_layers=3

echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="