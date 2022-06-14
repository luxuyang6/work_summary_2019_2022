import json



def process(data):
    datap = {}
    for k, g in data.items():
        n = 0
        for o in g['objects']:
            n += 1
        for r in g['relationships']:
            n += 1
        if n != 0:
            datap[k] = g 
    print('len(data)',len(data))
    print('len(datap)',len(datap))
    return datap

labels = json.load(open("labes.json"))

trn = json.load(open('Charades_v1_train_graph_o.json'))
tst = json.load(open('Charades_v1_test_graph_o.json'))
val = json.load(open('Charades_v1_val_graph_o.json'))
trnp = process(trn)
tstp = process(tst)
valp = process(val)
ftr = open('Charades_v1_train_graph_o.json','w')
fte = open('Charades_v1_test_graph_o.json','w')
fv = open('Charades_v1_val_graph_o.json','w')
json.dump(trnp,ftr)
json.dump(tstp,fte)
json.dump(valp,fv)
