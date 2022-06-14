import json


def process(objs, g):
    pairs = []
    gt_pairs = {}
    for id in g:
        vid = id.split('_')[0]
        if gt_pairs.get(vid) == None:
            gt_pairs[vid] = []
        for rel in g[id]['relationships']:
            gt_pairs[vid].append([rel['text'][0], rel['text'][2]])
            gt_pairs[vid].append([rel['text'][2], rel['text'][0]])

    for vid in objs:
        for i, obj1 in enumerate(objs[vid]):
            for j, obj2 in enumerate(objs[vid]):
                if i >= j:
                    continue
                label = int([obj1, obj2] in gt_pairs[vid])
                pairs.append((([obj1,obj2],vid),label))
    return pairs

objs_te = json.load(open('./results/model_epoch9.json'))
objs_tr = json.load(open('./results/model_epoch9_train.json'))
objs_va = json.load(open('./results/model_epoch9_val.json'))



te = json.load(open('../Charades/Charades_v1_test_graph_b.json')) 
tr = json.load(open('../Charades/Charades_v1_train_graph_b.json')) 
vl = json.load(open('../Charades/Charades_v1_val_graph_b.json')) 

oc_dict_te = process(objs_te, te)
oc_dict_tr = process(objs_tr, tr)
oc_dict_vl = process(objs_va, vl)

f_te = open('../Charades/Charades_pair_classifier_test.json','w') 
f_tr = open('../Charades/Charades_pair_classifier_train.json','w') 
f_vl = open('../Charades/Charades_pair_classifier_val.json','w')
json.dump(oc_dict_te, f_te)
json.dump(oc_dict_tr, f_tr)
json.dump(oc_dict_vl, f_vl)

