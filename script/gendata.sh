#!/bin/bash
model_name=gen_data
model_size=large
generate_type=$1
sample_num=20
batch_size=`expr 80 / $sample_num`
if [[ ${model_size} =~ "base" ]]; then
    batch_size=`expr $batch_size \* 2`
fi 
echo "batch size ${batch_size}"
dev_batch_size=64
top1000='data/marco/top1000-train'
dev_input_file='data/marco/top1000-dev'
warmup_proportion=0.2
eval_step_proportion=0.1
report_step=10
epoch=5
# eval_step_proportion=`echo "scale=5; 1/$epoch" | bc`
### 下面是永远不用改的
min_index=25
max_index=768
max_seq_len=160
collection='data/marco/collection.tsv'  
ernie_config_file=${model_size}/ernie_config.json  #ernie配置文件
vocab_file=${model_size}/vocab.txt  #ernie配置文件
warm_start_from=data/ernie_${model_size}.p  #ernie参数
qrels='data/marco/qrels.tsv'
query='data/marco/train.query.txt'
resource='data/concept.txt'
cpnet='data/conceptnet.en.pruned.graph'
pattern_path='data/matcher_patterns.json'
word2vec='data/GoogleNews-vectors-negative300.bin.gz'
ent_emb='data/glove.transe.sgd.ent.npy'
rel_emb='data/glove.transe.sgd.rel.npy'
output_dir=output
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
echo "=================start train ${OMPI_COMM_WORLD_RANK:-0}=================="
python -m paddle.distributed.launch \
    --log_dir ${log_dir} \
    ernie/generate_data.py \
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
    --num_gnn_layers=3 \
    --pattern_path=${pattern_path} \
    --word2vec=${word2vec} \
    --ent_emb=${ent_emb} \
    --rel_emb=${rel_emb} \
    --max_seq_len=${max_seq_len} \
    --model=ErnieWithGNNv2 \
    --generate_type=${generate_type}
echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="
# upload
tar -zcvf ${generate_type}.concept.tar.gz gen_data/
mv ${generate_type}.concept.tar.gz data/marco/