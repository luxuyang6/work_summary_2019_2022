import json


def process(objs, g):
    count = 0
    atts = []
    gt_atts = {}
    for id in g:
        # vid = id.split('_')[0]
        vid = id[0:-2]
        if gt_atts.get(vid) == None:
            gt_atts[vid] = []
        for obj in g[id]['objects']:
            for att in obj['attributes']:
                gt_atts[vid].append(obj['name']+att.split(" ")[0])

    for vid in objs:
        att_list = list(objs[vid])[:topk]
        att_list.append("person")
        for obj in att_list:
            for pred in kg_dict[obj]["attributes"][:topp]:
                label = 0
                if obj+pred[0] in gt_atts[vid] :
                    label = 1
                    count += 1
                atts.append((([obj, pred[0]],vid),label))
    print(count, len(atts))
    return atts


topk = 10
topp = 3 # 每个object选前topp个谓词

objs_te = json.load(open('./results/model_epoch5_test.json'))
objs_tr = json.load(open('./results/model_epoch5_train.json'))
objs_va = json.load(open('./results/model_epoch5_val.json'))

kg_dict = json.load(open('../ActivityNet/kg_dict_anet.json')) 

te = json.load(open('../ActivityNet/Anet_test_graph.json')) 
tr = json.load(open('../ActivityNet/Anet_train_graph.json')) 
vl = json.load(open('../ActivityNet/Anet_val_graph.json')) 

oc_dict_te = process(objs_te, te)
oc_dict_tr = process(objs_tr, tr)
oc_dict_vl = process(objs_va, vl)

f_te = open('../ActivityNet/Anet_att_classifier_test.json','w') 
f_tr = open('../ActivityNet/Anet_att_classifier_train.json','w') 
f_vl = open('../ActivityNet/Anet_att_classifier_val.json','w')
json.dump(oc_dict_te, f_te)
json.dump(oc_dict_tr, f_tr)
json.dump(oc_dict_vl, f_vl)

