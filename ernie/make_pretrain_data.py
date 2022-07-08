import spacy
import random
import numpy as np
import pandas as pd
from functools import reduce, partial
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



def make_data(cfg):
    collection = pd.read_csv(cfg['collection'],header=None,sep='\t')
    collection.columns=['pid','text']
    doc_number = len(collection)
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe('sentencizer')
    cnts = [0,0,0]  # random, next, previous
    for i in range(doc_number):
        doc = nlp(collection['text'][i])
        sents = list(doc.sents)
        sent_number = len(sents)
        for j,sent in enumerate(sents):
            if j and j<sent_number-1:
                srp_type = cnts.index(min(cnts))
                cnts[srp_type]+=1
                if srp_type==0:
                    pass
                elif srp_type==1:
                    pass
                else:
                    pass
            elif j and j==sent_number-1:
                srp_type = 0 if cnts[0]<cnts[2] else 2

            elif j==0 and sent_number>1:
                srp_type = 0 if cnts[0]<cnts[1] else 1
                cnts[srp_type]+=1
                if srp_type==0:
                    pass
                else:
                    pass
            elif j==0 and sent_number==1:
                srp_type = 0
                cnts[srp_type]+=1
            


