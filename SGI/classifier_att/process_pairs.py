import json
import pdb

# input
# {vid:[ [[obj01,obj02]: score0], [[obj11,obj12]: score1]... ]

# output
# [ [[[obj01,obj02,re01], vid], 0/1-label], [[[obj11,obj12,re11], vid], 0/1-label]... ]





def process(pairs, g):
    i = 0
    rels = []
    gt_rels = {}
    for id in g:
        vid = id.split('_')[0]
        if gt_rels.get(vid) == None:
            gt_rels[vid] = []
        for rel in g[id]['relationships']:
            gt_rels[vid].append(tuple(rel['text']))

    for vid in pairs:
        # if i % 100 == 0:
        #     print(i, vid)
        # i += 1
        for obj1 in pairs[vid]:
            for rel2 in ref_rels:
                # pdb.set_trace()
                rel3 = (obj1[0][0], rel2, obj1[0][1])
                label = int(rel3 in gt_rels[vid])
                rels.append((([obj1[0][0], rel2, obj1[0][1]], vid), label))
    return rels

ref_rels = json.load(open('../Charades/labes.json'))['rels']

pairs_te = json.load(open('./results/model_epoch9_test.json'))
pairs_tr = json.load(open('./results/model_epoch9_train.json'))
pairs_va = json.load(open('./results/model_epoch9_val.json'))

te = json.load(open('../Charades/Charades_v1_test_graph_b.json')) 
tr = json.load(open('../Charades/Charades_v1_train_graph_b.json')) 
vl = json.load(open('../Charades/Charades_v1_val_graph_b.json')) 

oc_dict_te = process(pairs_te, te)
oc_dict_tr = process(pairs_tr, tr)
oc_dict_vl = process(pairs_va, vl)

f_te = open('../Charades/Charades_rel_classifier_test.json','w') 
f_tr = open('../Charades/Charades_rel_classifier_train.json','w') 
f_vl = open('../Charades/Charades_rel_classifier_val.json','w')
json.dump(oc_dict_te, f_te)
json.dump(oc_dict_tr, f_tr)
json.dump(oc_dict_vl, f_vl)

