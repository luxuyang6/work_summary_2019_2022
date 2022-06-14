import json
trn = json.load(open('Charades_v1_train_graph_o.json'))
val_key = list(trn.keys())[-2000:]
val = {}
for key in val_key:
    val[key] = trn[key]
    trn.pop(key)
ft = open('Charades_v1_train_graph_o.json','w')
fv = open('Charades_v1_val_graph_o.json','w')
json.dump(trn,ft)
json.dump(val,fv)
