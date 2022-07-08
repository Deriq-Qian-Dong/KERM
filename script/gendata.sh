#!/bin/bash
model_name=$1
queue=$2
model_size=$3
sample_num=20
if [[ ${queue} =~ "v100" ]]; then
    batch_size=6
fi 
if [[ ${queue} =~ "p40" ]]; then
    batch_size=6
fi 
if [[ ${queue} =~ "a100" ]]; then
    batch_size=`expr 80 / $sample_num`
fi 
if [[ ${model_size} =~ "base" ]]; then
    batch_size=`expr $batch_size \* 2`
fi 
echo "batch size ${batch_size}"
dev_batch_size=64
top1000='data/train-res.top1000'
# top1000='data/train.qidpid.gz'  #训练数据
# top1000='data/treccar/train.qidpid.tsv'
# top1000='data/RocketQAv2/data_train/marco_joint.rand128+aug128'
# dev_input_file='data/dev.bm25.gz'  #测试数据
# dev_input_file='data/dev.top1000.bm25_tuned.csv'
# dev_input_file='data/dev.top1000.bm25_tuned.csv'
# dev_input_file='data/eval_39.60_98.88.top1000.tsv'
dev_input_file='data/dev_39.48_98.88.top1000.tsv'
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
rm -rf /etc/pip.conf
cp pip.conf /etc/pip.conf
pip install networkx
pip install data/networkx-2.6.3-py3-none-any.whl
pip install pgl
pip install data/pgl-2.1.5-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install spacy
pip install data/spacy-3.2.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install nltk
pip install gensim
pip install data/gensim-4.1.2-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
python -m spacy download en_core_web_sm
pip install data/en_core_web_sm-3.2.0-py3-none-any.whl
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
    --model=ErnieWithGNNv2
echo "=================done train ${OMPI_COMM_WORLD_RANK:-0}=================="
# upload
echo "Starting uploading file to HDFS"
tar -zcvf /root/paddlejob/workspace/env_run/output.tar.gz gen_data/
${hdfs_cmd} -mkdir /user/sasd-adv/diaoyan/user/modelzoo/${model_name}
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/output.tar.gz /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/ernie /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
${hdfs_cmd} -put  /root/paddlejob/workspace/env_run/script /user/sasd-adv/diaoyan/user/modelzoo/${model_name}/
echo "Done uploading file to HDFS"
