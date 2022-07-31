# KERM

Code for KERM: Incorporating Explicit Knowledge in Pre-trained Language Models for Passage Re-ranking, accepted at SIGIR 2022.

### Dependencies
> networkx==2.6.3

> paddlepaddle-gpu==2.1.0

> pgl==2.1.5

> spacy==3.2.0

> gensim==4.1.2

> At least 4*Tesla A100(40GB)
### Data preparation
1. The ConceptNet-related resources used in KERM can be downloaded from [here](https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws), which is provided by [MHGRN](https://github.com/INK-USC/MHGRN). Placed in "data/". 
2. MARCO and TREC 2019DL could be downloaded from [here](https://microsoft.github.io/msmarco/). Placed in "data/dataset_name".
3. The bio-medical dataset Ohsumed is available at [here](http://disi.unitn.it/moschitti/corpora.htm). Placed in "data/dataset_name".
4. The word2vec embedding could be downloaded from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g). Placed in "data/".
5. The parameters of ERNIE-2.0 could be downloaded from [here](https://github.com/PaddlePaddle/ERNIE/blob/3fb0b4911d5be66ef157df3d46d046e16ffc7b36/README.eng.md#3-download-pretrained-models-optional), including both base and large model. Placed in "data/".
6. The top1000-train passages for training queries could be retrieved by [this](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQA_NAACL2021#dual-encoder-inference), from which the hard negatives are sampled based on the Dense Passage Retrieval (DPR) in [RocketQA](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQA_NAACL2021). Notably, the [top1000-dev](https://www.dropbox.com/s/5pqpcnlzlib2b3a/run.bm25.dev.small.tsv.gz?dl=1) for dev queries is obtained from this [repo](https://github.com/castorini/duobert), which is widely-used in previous works. After obtaining the top1000 data, please convert them into the flowing format: (Placed in "data/dataset_name".) 
>top1000-train
>>query_id  passage_id  index   score
>>>121352  2912791 1       131.90326

>>>121352  7282917 2       131.07689

>>>121352  7480161 3       130.65248

>top1000-dev
>>query_id  passage_id  query_text  -   passage_text    label
>>>188714  2133570 foods and supplements to lower blood sugar      -       A healthy diet is essential to reversing prediabetes. There are no foods, herbs, drinks, or supplements that lower blood sugar. Only medication and exercise can. But there are things you can eat and drink that are low on the glycemic index (GI). This means these foods wonât raise your blood sugar and may help you avoid a blood sugar spike. 0

>>>188714  4321742 foods and supplements to lower blood sugar      -       Ohio State University, researchers saw insulin levels drop 23 percent and blood sugar levels drop 29 percent in patients who took a 1,000-mg dose of the herb. Amazing! These are just a few of the natural foods and supplements that will lower your blood sugar level naturally. One thing that is very important is that you keep your health care provider up to date on any supplements that you will be utilizing as a natural way to lower your blood sugar.     0

>>>188714  4321745 foods and supplements to lower blood sugar      -       Food And Supplements That Lower Blood Sugar Levels. Cinnamon: Researchers are finding that cinnamon reduces blood sugar levels naturally when taken daily. If you absolutely love cinnamon you can sprinkle the recommended six grams of cinnamon on your food throughout the day to achieve the desired effect.      1
### Data generation
For the efficiency of training, we first generate the data used in our work once for all. We take the MARCO and KERM-large for example.
1. Generate the data for knowledge-enhanced pre-training:
>sh script/batch_submit_genpretrain.sh
2. Generate the data for training and evaluation:
>sh script/gendata.sh train

>sh script/gendata.sh eval

### Knowledge-enhanced pre-training & finetune
To better training the KERM, we first continuous pre-training the ERNIE-large to warm up the parameters of GMN:
>sh script/pretrain_large.sh

>sh script/large_finetune.sh

### Acknowledgement
Some snippets of the codes are borrowed from [MHGRN](https://github.com/INK-USC/MHGRN), [ERNIE](https://github.com/PaddlePaddle/ERNIE) and [ERNIE-THU](https://github.com/thunlp/ERNIE).
To cite this paper, use the following BibTex:
> @article{dong2022incorporating,
  title={Incorporating Explicit Knowledge in Pre-trained Language Models for Passage Re-ranking},
  author={Dong, Qian and Liu, Yiding and Cheng, Suqi and Wang, Shuaiqiang and Cheng, Zhicong and Niu, Shuzi and Yin, Dawei},
  journal={arXiv preprint arXiv:2204.11673},
  year={2022}
}
