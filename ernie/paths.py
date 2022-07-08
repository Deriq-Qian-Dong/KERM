import sys
import json
import numpy as np
import networkx as nx
import tokenization
from spacy.matcher import Matcher
import spacy
import gensim
import pgl
import re
class ConceptGraphBuilder:
    def __init__(self,cfg):
        self.cfg = cfg
        self.load_resources(cfg['resource'])
        self.load_cpnet(cfg['cpnet'])
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        self.nlp.add_pipe('sentencizer')
        self.matcher = Matcher(self.nlp.vocab)
        with open(cfg['pattern_path'], "r", encoding="utf8") as fin:
            all_patterns = json.load(fin)
        for concept, pattern in all_patterns.items():
            self.matcher.add(concept, [pattern])
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(cfg['word2vec'], binary=True)
        self.stopwords=['i',
                        'me',
                        'my',
                        'myself',
                        'we',
                        'our',
                        'ours',
                        'ourselves',
                        'you',
                        "you're",
                        "you've",
                        "you'll",
                        "you'd",
                        'your',
                        'yours',
                        'yourself',
                        'yourselves',
                        'he',
                        'him',
                        'his',
                        'himself',
                        'she',
                        "she's",
                        'her',
                        'hers',
                        'herself',
                        'it',
                        "it's",
                        'its',
                        'itself',
                        'they',
                        'them',
                        'their',
                        'theirs',
                        'themselves',
                        'what',
                        'which',
                        'who',
                        'whom',
                        'this',
                        'that',
                        "that'll",
                        'these',
                        'those',
                        'am',
                        'is',
                        'are',
                        'was',
                        'were',
                        'be',
                        'been',
                        'being',
                        'have',
                        'has',
                        'had',
                        'having',
                        'do',
                        'does',
                        'did',
                        'doing',
                        'a',
                        'an',
                        'the',
                        'and',
                        'but',
                        'if',
                        'or',
                        'because',
                        'as',
                        'until',
                        'while',
                        'of',
                        'at',
                        'by',
                        'for',
                        'with',
                        'about',
                        'against',
                        'between',
                        'into',
                        'through',
                        'during',
                        'before',
                        'after',
                        'above',
                        'below',
                        'to',
                        'from',
                        'up',
                        'down',
                        'in',
                        'out',
                        'on',
                        'off',
                        'over',
                        'under',
                        'again',
                        'further',
                        'then',
                        'once',
                        'here',
                        'there',
                        'when',
                        'where',
                        'why',
                        'how',
                        'all',
                        'any',
                        'both',
                        'each',
                        'few',
                        'more',
                        'most',
                        'other',
                        'some',
                        'such',
                        'no',
                        'nor',
                        'not',
                        'only',
                        'own',
                        'same',
                        'so',
                        'than',
                        'too',
                        'very',
                        's',
                        't',
                        'can',
                        'will',
                        'just',
                        'don',
                        "don't",
                        'should',
                        "should've",
                        'now',
                        'd',
                        'll',
                        'm',
                        'o',
                        're',
                        've',
                        'y',
                        'ain',
                        'aren',
                        "aren't",
                        'couldn',
                        "couldn't",
                        'didn',
                        "didn't",
                        'doesn',
                        "doesn't",
                        'hadn',
                        "hadn't",
                        'hasn',
                        "hasn't",
                        'haven',
                        "haven't",
                        'isn',
                        "isn't",
                        'ma',
                        'mightn',
                        "mightn't",
                        'mustn',
                        "mustn't",
                        'needn',
                        "needn't",
                        'shan',
                        "shan't",
                        'shouldn',
                        "shouldn't",
                        'wasn',
                        "wasn't",
                        'weren',
                        "weren't",
                        'won',
                        "won't",
                        'wouldn',
                        "wouldn't"]
        self.blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                        "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                        "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                        ]+self.stopwords)
    def get_span(self, sent):
        sent, shift = sent
        sent = sent.lower()
        # sent = re.sub(r'[^a-z ]+', ' ', sent)
        sent = sent.replace("-", "_")
        spans = []
        tokens = sent.split(" ")
        token_num = len(tokens)
        itv = []
        for length in range(1, 5):
            for i in range(token_num-length+1):
                span = "_".join(tokens[i:i+length])
                span = list(self.lemmatize(span))[0]
                if span not in self.blacklist and span in self.concept2id and span not in spans:
                    spans.append(span)
                    itv.append((span,i,i+length))
        itv = self.removeCoveredIntervals(itv)
        return [(i[0],i[1]+shift,i[2]+shift) for i in itv]
    def get_edge(self, src_concept, tgt_concept):
        rel_list = self.cpnet[src_concept][tgt_concept]  # list of dicts
        seen = set()
        res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
        return res
    def find_paths_qd_concept_pair(self, source: str, target: str, ifprint=False):
        s = self.concept2id[source]
        t = self.concept2id[target]
        if s not in self.cpnet_simple.nodes() or t not in self.cpnet_simple.nodes():
            return []
        all_path = []
        top_paths = 10
        try:
            for p in nx.shortest_simple_paths(self.cpnet_topk, source=s, target=t):
                if len(p) > 4 or len(all_path) >= top_paths:  # top 10 paths
                    break
                if len(p) >= 2:  # skip paths of length 1
                    all_path.append(p)
        except nx.exception.NetworkXNoPath:
            return []
        pf_res = []
        for p in all_path:
            rl = []
            for src in range(len(p) - 1):
                src_concept = p[src]
                tgt_concept = p[src + 1]

                rel_list = self.get_edge(src_concept, tgt_concept)
                rl.append(rel_list)
                if ifprint:
                    rel_list_str = []
                    for rel in rel_list:
                        if rel < len(self.id2relation):
                            rel_list_str.append(self.id2relation[rel])
                        else:
                            rel_list_str.append(self.id2relation[rel - len(self.id2relation)] + "*")
                    print(self.id2concept[src_concept], "----[%s]---> " % ("/".join(rel_list_str)), end="")
                    if src + 1 == len(p) - 1:
                        print(self.id2concept[tgt_concept], end="")
            if ifprint:
                print()
            pf_res.append({"path": p, "rel": rl})
        return pf_res
    def lemmatize(self, concept):
        doc = self.nlp(concept.replace("_", " "))
        lcs = set()
        lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
        return lcs
    def removeCoveredIntervals(self, intervals):
        intervals.sort(key=lambda x:(x[1], -x[2]))
        dp=[]
        for itv in intervals:
            if not dp or dp[-1][2]<itv[2]:
                dp.append(itv)
        return dp
    def ground_mentioned_concepts(self, s):
        s = s.lower()
        doc = self.nlp(s)
        matches = self.removeCoveredIntervals(self.matcher(doc))
        mentioned_concepts = set()
        span_to_concepts = {}
        for match_id, start, end in matches:
            span = doc[start:end].text  # the matched span
            original_concept = self.nlp.vocab.strings[match_id]
            original_concept_set = set()
            original_concept_set.add(original_concept)
            if len(original_concept.split("_")) == 1:
                original_concept_set.update(self.lemmatize(self.nlp.vocab.strings[match_id]))

            if span not in span_to_concepts:
                span_to_concepts[span] = set()

            span_to_concepts[span].update(original_concept_set)

        for span, concepts in span_to_concepts.items():
            concepts_sorted = list(concepts)
            concepts_sorted.sort(key=len)
            shortest = concepts_sorted[0:3]

            for c in shortest:
                if c in self.blacklist:
                    continue
                lcs = self.lemmatize(c)
                intersect = lcs.intersection(shortest)
                if len(intersect) > 0:
                    mentioned_concepts.add(list(intersect)[0])
                else:
                    mentioned_concepts.add(c)

            exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
            mentioned_concepts.update(exact_match)
        return mentioned_concepts
    def get_emb(self, sent):
        return np.mean(np.array([self.word2vec[w] for w in sent.split(" ") if w in self.word2vec]),axis=0)
    def get_topk_sents(self, qry, doc, k):
        qemb=self.get_emb(qry)
        d=self.nlp(doc)
        ret = []
        shift = 0
        for s in d.sents:
            sent = re.sub(r'[^a-z ]+', ' ', s.text.lower())
            semb=self.get_emb(sent)
            if not np.isnan(semb).all():
                sim=np.dot(qemb,semb)
                if type(sim) is np.float32:
                    ret.append((sim, s.text.lower(), shift))
            shift+=len(s.text.split(" "))
        ret.sort(key=lambda x:x[0],reverse=True)
        return [(r[1], r[2]) for r in ret[:k]]
    def get_graph(self, qry, doc, is_print=False):
        topk_sents_and_shift = self.get_topk_sents(qry, doc, self.cfg['topk_sents'])
        qry = qry.lower()
        doc = doc.lower()
        qry_concepts_and_shift = self.get_span((qry,0))
        doc_concepts_and_shift = [('ssss',0,1)]
        for sent_and_shift in topk_sents_and_shift:
            doc_concepts_and_shift += self.get_span(sent_and_shift)
        doc_concepts_and_shift = list(set(doc_concepts_and_shift))
        paths=[]
        for q in qry_concepts_and_shift:
            for d in doc_concepts_and_shift:
                paths+=self.find_paths_qd_concept_pair(q[0],d[0],is_print)
        nodes = {}
        qry_nodes = []
        doc_nodes = []
        edges = []
        edges_type = []
        for path in paths:
            for i,node in enumerate(path['path']):
                if node not in nodes:
                    nodes[node] = len(nodes)
                if i:
                    prev_node = nodes[path['path'][i-1]]
                    edges.append((prev_node, nodes[node]))
                    edges_type.append(path['rel'][i-1][0]%17)  
        q_spans = []
        d_spans = []
        qry_concepts_to_shift={}
        for c,s,e in qry_concepts_and_shift:
            qry_concepts_to_shift[self.concept2id[c]]=(s,e)
        doc_concepts_to_shift = {}
        for c,s,e in doc_concepts_and_shift:
            doc_concepts_to_shift[self.concept2id[c]] = (s,e)
        for path in paths:
            q_node = path['path'][0]
            d_node = path['path'][-1]
            if nodes[q_node] not in qry_nodes:
                qry_nodes.append(nodes[q_node])
                q_spans.append(qry_concepts_to_shift[q_node])
            if nodes[d_node] not in doc_nodes:
                doc_nodes.append(nodes[d_node])
                d_spans.append(doc_concepts_to_shift[d_node])
        
        num_nodes = len(nodes)    
        nodes_feature = self.ent_emb[list(nodes)]
        edges_feature = self.rel_emb[edges_type]
        graph = pgl.Graph(num_nodes=num_nodes,
                edges=edges,
                node_feat={"feature": nodes_feature},
                edge_feat={"edge_feature":edges_feature})
        return graph,q_spans,d_spans,qry_nodes,doc_nodes


    def load_resources(self,cpnet_vocab_path):
        with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
            self.id2concept = [w.strip() for w in fin]
        self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
        self.id2relation = [
                            'antonym',
                            'atlocation',
                            'capableof',
                            'causes',
                            'createdby',
                            'isa',
                            'desires',
                            'hassubevent',
                            'partof',
                            'hascontext',
                            'hasproperty',
                            'madeof',
                            'notcapableof',
                            'notdesires',
                            'receivesaction',
                            'relatedto',
                            'usedfor',
                            ]
        self.relation2id = {r: i for i, r in enumerate(self.id2relation)}
    def load_cpnet(self,cpnet_graph_path):
        cpnet = nx.read_gpickle(cpnet_graph_path)
        cpnet.add_edge(self.concept2id['cccc'],self.concept2id['ssss'],weight=1.0,rel=15)
        cpnet_simple = nx.Graph()
        self.ent_emb = np.load(self.cfg['ent_emb'])
        self.ent_emb = np.concatenate([self.ent_emb, np.zeros((2,100))]).astype("float32")
        self.rel_emb = np.load(self.cfg['rel_emb']).astype("float32")
        for u, v, data in cpnet.edges(data=True):
            rid = data['rel']%17
            w = np.inner(self.ent_emb[u], self.ent_emb[v])+np.inner(self.ent_emb[u], self.rel_emb[rid])+np.inner(self.rel_emb[rid], self.ent_emb[v])
            w = 1/abs(w)
            if not cpnet_simple.has_edge(u, v):
                cpnet_simple.add_edge(u, v, weight=w)
        topk={}
        max_neighbor=50 #消融改了这个
        # max_neighbor = 8000000
        for u in cpnet_simple:
            topk[u] = []
            for v,data in cpnet_simple[u].items():
                topk[u].append((data['weight'],v))
            topk[u].sort(key=lambda x:x[0])
            topk[u]=topk[u][:max_neighbor]
        cpnet_topk = nx.DiGraph()
        for u in topk:
            for w,v in topk[u]:
                if not cpnet_topk.has_edge(u,v):
                    cpnet_topk.add_edge(u,v,weight=w)
        self.cpnet_topk = cpnet_topk
        self.cpnet_simple = cpnet_simple
        self.cpnet = cpnet

  
class GraphBuilder:
    def __init__(self, cfg):
        self.load_resources(cfg['resource'])
        self.load_cpnet(cfg['cpnet'])
        self.tokenizer = tokenization.FullTokenizer(cfg['vocab_file'])
        self.cfg = cfg
     
    def load_resources(self,cpnet_vocab_path):
        with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
            self.id2concept = [w.strip() for w in fin]
        self.concept2id = {w: i for i, w in enumerate(self.id2concept)}

        self.id2relation = merged_relations=['atlocation',
                                             'capableof',
                                             'causes',
                                             'createdby',
                                             'desires',
                                             'hasproperty',
                                             'madeof',
                                             'notcapableof',
                                             'notdesires',
                                             'partof',
                                             'usedfor',
                                             'receivesaction']
        self.relation2id = {r: i for i, r in enumerate(self.id2relation)}


    def load_cpnet(self,cpnet_graph_path):
        global cpnet, cpnet_simple
        cpnet = nx.read_gpickle(cpnet_graph_path)
        cpnet_simple = nx.Graph()
        for u, v, data in cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        self.cpnet_simple = cpnet_simple
        self.cpnet = cpnet



    def get_graph(self, query_tokens_id, doc_tokens_id, is_print=False):
        edges = []
        nodes_num = self.cfg['q_max_seq_len']+self.cfg['p_max_seq_len']
        nodes_tokens = []
        edges_type = []
        query_tokens = self.tokenizer.convert_ids_to_tokens(query_tokens_id)
        doc_tokens = self.tokenizer.convert_ids_to_tokens(doc_tokens_id)
        for q_tok_id, q_token in enumerate(query_tokens):
            if self.concept2id.get(q_token,-1)!=-1:
                for d_tok_id, d_token in enumerate(doc_tokens):
                    if self.concept2id.get(d_token,-1)!=-1:
                        paths = self.find_paths_qd_concept_pair(q_token, d_token, is_print)
                        d_tok_id+=self.cfg['q_max_seq_len']
                        for p in paths:
                            if len(p['path'])==2:
                                edges.append((q_tok_id, d_tok_id))
                                edges_type.append(p['rel'][0][0])
                            else:
                                edges.append((q_tok_id, nodes_num))
                                edges_type.append(p['rel'][0][0])

                                for i,node in enumerate(p['path'][1:-2]):
                                    edges.append((nodes_num, nodes_num+1))
                                    edges_type.append(p['rel'][i+1][0])
                                    nodes_tokens.append(node)
                                    nodes_num+=1

                                edges.append((nodes_num, d_tok_id))
                                edges_type.append(p['rel'][-1][0])

                                nodes_tokens.append(p['path'][-2])
                                nodes_num+=1
        return edges, edges_type, nodes_tokens,nodes_num

    def get_edge(self, src_concept, tgt_concept):
        rel_list = self.cpnet[src_concept][tgt_concept]  # list of dicts
        seen = set()
        res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
        return res

    def find_paths_qd_concept_pair(self,source: str, target: str, ifprint=False):
        s = self.concept2id[source]
        t = self.concept2id[target]

        if s not in self.cpnet_simple.nodes() or t not in self.cpnet_simple.nodes():
            return []
        all_path = []
        try:
            for p in nx.shortest_simple_paths(self.cpnet_simple, source=s, target=t):
                if len(p) > 4 or len(all_path) >= 20:  # top 20 paths
                    break
                if len(p) >= 2:  # skip paths of length 1
                    all_path.append(p)
        except nx.exception.NetworkXNoPath:
            return []

        pf_res = []
        for p in all_path:
            # print([id2concept[i] for i in p])
            rl = []
            for src in range(len(p) - 1):
                src_concept = p[src]
                tgt_concept = p[src + 1]

                rel_list = self.get_edge(src_concept, tgt_concept)
                rl.append(rel_list)
                if ifprint:
                    rel_list_str = []
                    for rel in rel_list:
                        if rel < len(self.id2relation):
                            rel_list_str.append(self.id2relation[rel])
                        else:
                            rel_list_str.append(self.id2relation[rel - len(self.id2relation)] + "*")
                    print(self.id2concept[src_concept], "----[%s]---> " % ("/".join(rel_list_str)), end="")
                    if src + 1 == len(p) - 1:
                        print(self.id2concept[tgt_concept], end="")
            if ifprint:
                print()

            pf_res.append({"path": [self.tokenizer.vocab[self.id2concept[concept_id]] for concept_id in p], "rel": rl})
        return pf_res



class BatchedGraphBuilder(GraphBuilder):
    def __init__(self, cfg):
        GraphBuilder.__init__(self, cfg)

    def get_batched_graph(self, pair_input_ids, is_print=False):
        # pair_input_ids:[batch_size, seq_len]
        batch_size = pair_input_ids.shape[0]
        seq_len = pair_input_ids.shape[1]
        edges = []
        nodes_num = batch_size*seq_len
        nodes_tokens = []
        edges_type = []
        batched_query_tokens_id = []
        batched_doc_tokens_id = []
        for sample_id in range(batch_size):
            sep_id = np.where(pair_input_ids[sample_id]==102)[0][0]
            batched_query_tokens_id.append(pair_input_ids[sample_id][:sep_id])
            batched_doc_tokens_id.append(pair_input_ids[sample_id][sep_id:])
            edges.append((seq_len*sample_id, seq_len*sample_id+sep_id))
            edges_type.append(self.cfg['edge_num']-1)
        for batch_i,query_tokens_id in enumerate(batched_query_tokens_id):
            doc_tokens_id = batched_doc_tokens_id[batch_i]
            query_tokens = self.tokenizer.convert_ids_to_tokens(query_tokens_id.reshape(-1))
            doc_tokens = self.tokenizer.convert_ids_to_tokens(doc_tokens_id.reshape(-1))
            # print(query_tokens,len(query_tokens))
            # print(doc_tokens,len(doc_tokens))
            for q_tok_id, q_token in enumerate(query_tokens):
                if self.concept2id.get(q_token,-1)!=-1:
                    q_tok_id = q_tok_id + seq_len*batch_i
                    for d_tok_id, d_token in enumerate(doc_tokens):
                        if self.concept2id.get(d_token,-1)!=-1:
                            paths = self.find_paths_qd_concept_pair(q_token, d_token, False)
                            d_tok_id = d_tok_id + seq_len*batch_i + len(query_tokens)
                            # if paths:
                            #     print(q_tok_id,d_tok_id,q_token,d_token)
                            for p in paths:
                                if len(p['path'])==2:
                                    edges.append((q_tok_id, d_tok_id))
                                    edges_type.append(p['rel'][0][0])
                                else:
                                    edges.append((q_tok_id, nodes_num))
                                    edges_type.append(p['rel'][0][0])

                                    for i,node in enumerate(p['path'][1:-2]):
                                        edges.append((nodes_num, nodes_num+1))
                                        edges_type.append(p['rel'][i+1][0])
                                        nodes_tokens.append(node)
                                        nodes_num+=1

                                    edges.append((nodes_num, d_tok_id))
                                    edges_type.append(p['rel'][-1][0])

                                    nodes_tokens.append(p['path'][-2])
                                    nodes_num+=1
                                    if nodes_num>=1.8*(batch_size*seq_len):
                                        return (edges, edges_type, nodes_tokens,nodes_num)
        return (edges, edges_type, nodes_tokens,nodes_num)

