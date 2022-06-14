import json


# data format:
# oc_dict[vid][class_label][bool]


def process(g):
    oc_dict = {}
    out_dict = {}
    for n,id in enumerate(g):
        vid = id.split('_')[0]
        if oc_dict.get(vid) == None:
            oc_dict[vid] = {}
            for obj in outdict[vid]:
                oc_dict[vid][obj] = 0
        for i,o in enumerate(g[id]['objects']):
            oc_dict[vid][o['name']] = 1
    
    for fid in watch_dict:
        vid = fid[:5]
        if out_dict.get(vid) == None:
            outset = set()
            for obj in oc_dict[vid]:
                if oc_dict[vid][obj] == 1:
                    outset.add(obj)
            out_dict[vid] = list(outset)
        outs = set()
        for obj in watch_dict[fid][:20]:
            if oc_dict[vid][obj] == 1:
                outs.add(obj)
        out_dict[fid] = list(outs)
        

    return out_dict
    # return oc_dict


te = json.load(open('Charades_v1_test_graph_b.json')) 
# tr = json.load(open('Charades_v1_train_graph_b.json')) 
# vl = json.load(open('Charades_v1_val_graph_b.json')) 
outdict = json.load(open('outdict2.json'))
watch_dict = json.load(open('QMHK8-2.json'))

oc_dict_te = process(te)
# oc_dict_tr = process(tr)
# oc_dict_vl = process(vl)

f_te = open('QMHK8-3.json','w') 
# f_te = open('Charades_object_classifier_test.json','w') 
# f_tr = open('Charades_object_classifier_train.json','w') 
# f_vl = open('Charades_object_classifier_val.json','w')
json.dump(oc_dict_te, f_te)
# json.dump(oc_dict_tr, f_tr)
# json.dump(oc_dict_vl, f_vl)