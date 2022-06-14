import json
import csv
import re
import spacy

def read_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        vid = []
        description = []
        for row in reader:
            _vid = row['id']
            #_description = row['descriptions'].split('.')
            _description = re.split(r'[\.;]',row['descriptions'].lower().replace(',',' ,'))
            _description = [re.split(r'[ /]',sent)+['.'] for sent in _description if sent.split() != []]
            _description = [[word for word in sent if word != ""] for sent in _description]
            _description = [' '.join(sent) for sent in _description]
            vid.append(_vid)
            description.append(_description)
        raw_data_dict = {'vid':vid, 'description':description}
        print('len(data)',len(raw_data_dict['vid']))

    return raw_data_dict

def preprocess(raw_data):
    descriptions = raw_data['description']
    processed_des = []
    for des in descriptions:
        des_ = []
        for sent in des:
            sent_split = []
            doc = nlp(sent)
            for token in doc:
                sent_split.append(token.text)
            pre_pos = 'sos'
            for index in range(len(doc)):
                i = len(doc)- index - 1

                # if doc[i].pos_ == 'VERB' and doc[i-1].pos_ != 'AUX' and doc[i-1].pos_ != 'PRON':
                # if doc[i].pos_ == 'VERB' and doc[i-1].pos_ != 'AUX':
                if doc[i].pos_ == 'VERB':
                    # if doc[i-1].pos_ == 'NOUN' and doc[i].text[-3:] != 'ing':
                    # if doc[i-1].pos_ == 'NOUN' and doc[i].text[-3:] != 'ing':
                    #     continue
                    sent_split[i] = doc[i].lemma_
                    if doc[i-1].pos_ == 'PUNCT':
                        sent_split.insert(i,'the person')
                    else:
                        sent_split.insert(i,', the person')
            des_.append(' '.join(sent_split))
        processed_des.append(des_)
    return {'vid':raw_data['vid'], 'description':processed_des, 'origin_description':descriptions}


if __name__ == "__main__":
    mode = "train"
    # mode = "train"
    nlp = spacy.load("en_core_web_lg")
    if mode == "train":
        raw_data = read_csv('./Charades_v1_train.csv')
        f_out = open('./Charades_v1_train.json','w')
    else:
        raw_data = read_csv('./Charades_v1_test.csv')
        f_out = open('./Charades_v1_test.json','w')
    processed_data = preprocess(raw_data)
    json.dump(processed_data, f_out)