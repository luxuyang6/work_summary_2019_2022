import json
import nltk
from nltk.stem import PorterStemmer
import spacy



def convert_text(pre_input, java_input):
    pre_input.pop('description')
    vid = pre_input['vid']
    output_graph = []
    output_dict = {}
    for j, data in enumerate(java_input):
        for i, sent in enumerate(data):
            objects = sent['objects']
            relationships = sent['relationships']
            attributes = sent['attributes']
            map_list = []
            del_list = []
            map_dict = {}
            id = 0
            for idx,obj in enumerate(objects):
                objects[idx]['object_id'] = id
                obj['names'] = objects[idx]['name'] = objects[idx]['names'][0]
                if map_dict.get(obj['names']) == None:
                    map_dict[obj['names']] = id
                    id += 1
                else:
                    del_list.append(idx)
                map_list.append(map_dict[obj['names']])
                objects[idx].pop('names')
                objects[idx]['attributes'] = []
            for idx in del_list[::-1]:
                del objects[idx]
            rel_dict = {}
            id = 0
            relations = []
            for idx, rel in enumerate(relationships):
                rel['relationship_id'] = id
                rel['name'] = rel['predicate']
                rel['subject_id'] = map_list[rel['subject']]
                rel['object_id'] = map_list[rel['object']]
                if rel['subject_id'] == rel['object_id']:
                    continue
                rel.pop('predicate')
                rel.pop('object')
                rel.pop('subject')
                if rel_dict.get((rel['name'],rel['subject_id'],rel['object_id'])) == None:
                    rel_dict[((rel['name'],rel['subject_id'],rel['object_id']))] = id
                    id += 1
                relations.append(rel)
            # print(map_list)
            for att in attributes:
                subj = map_list[att['subject']]
                objects[subj]['attributes'].append(att['attribute'])
            data[i]['objects'] = objects
            data[i]['relationships'] = relations
            data[i].pop('id')
            data[i].pop('attributes')
            data[i].pop('url')
        output_dict[vid[j]] = data
    return output_dict




def convert_video():
    # 获得词频dict
    # wordlist = json.load(open('wordlist.json'))
    # word_dict ={}
    # ps = PorterStemmer()
    # word_num = 0
    # for word in wordlist:
    #     word_num += word[1]
    # for word in wordlist:
    #     word_dict[ps.stem(word[0])] = word[1]/word_num
    
    return word_dict

                


if __name__ == "__main__":
    mode = "test"
    # mode = "train"
    nlp = spacy.load("en_core_web_lg")

    if mode == "train":
        pre_input = json.load(open("./Charades_v1_train.json"))
        java_input = json.load(open("./stanford-corenlp-full-2015-12-09/train.json"))
        text_out = open('./Charades_v1_train_graph.json','w')
    else:
        pre_input = json.load(open("./Charades_v1_test.json"))
        java_input = json.load(open("./stanford-corenlp-full-2015-12-09/test.json"))
        text_out = open('./Charades_v1_test_graph.json','w')
    text_dict = convert_text(pre_input, java_input)
    json.dump(text_dict, text_out)