import json
import pdb


def process(objs, g):
    count = 0
    rels = []
    gt_rels = {}
    for id in g:
        vid = id.split('_')[0]
        if gt_rels.get(vid) == None:
            gt_rels[vid] = []
        for rel in g[id]['relationships']:
            gt_rels[vid].append(rel['text'][0]+rel['text'][1].split(" ")[0]+rel['text'][2])

    for vid in objs:
        for obj in list(objs[vid])[:topk]:
            for pred in kg_dict[obj]["relations"][:topp]:
                label = 0
                for subj in ["person", "man", "woman", "boy", "girl"]:
                    if subj+pred[0]+obj in gt_rels[vid] :
                        label = 1
                        count += 1
                        break
                rels.append(((["person", pred[0], obj],vid),label))
    print(count, len(rels))
    return rels

topk = 10
topp = 3 # 每个object选前topp个谓词

objs_te = json.load(open('./results/model_epoch5_test.json'))
objs_tr = json.load(open('./results/model_epoch5_train.json'))
objs_va = json.load(open('./results/model_epoch5_val.json'))

kg_dict = json.load(open('../Charades/kg_dict.json')) 

te = json.load(open('../Charades/Charades_v1_test_graph_b.json')) 
tr = json.load(open('../Charades/Charades_v1_train_graph_b.json')) 
vl = json.load(open('../Charades/Charades_v1_val_graph_b.json')) 

oc_dict_te = process(objs_te, te)
oc_dict_tr = process(objs_tr, tr)
oc_dict_vl = process(objs_va, vl)

f_te = open('../Charades/Charades_rel_classifier_test.json','w') 
f_tr = open('../Charades/Charades_rel_classifier_train.json','w') 
f_vl = open('../Charades/Charades_rel_classifier_val.json','w')
json.dump(oc_dict_te, f_te)
json.dump(oc_dict_tr, f_tr)
json.dump(oc_dict_vl, f_vl)

