import json


labels = json.load(open("labes.json"))
outdict = json.load(open("outdict2.json"))

len_obj = 0
len_att = 0
len_rel = 0

def process(data):

    global len_obj
    global len_att
    global len_rel
    datap = {}
    for k, g in data.items():
        n = 0
        vid = k.split('_')[0]
        if vid == 'NSKMC' or vid == 'K48CQ':
            continue
        del_objs = []
        objects = g['objects']
        objects = list(filter(lambda x: x['name'] in outdict[vid], objects))
        len_obj += len(objects)
        # objects = list(filter(lambda x: x['name'] in labels['objs'], objects))
        for o in g['objects']:
            if o not in objects:
                del_objs.append(o['object_id'])
        for i,o in enumerate(objects):
            attributes = o['attributes']
            attributes = list(filter(lambda x: x in labels['atts'], attributes))
            len_att += len(attributes)
            objects[i]['attributes'] = attributes
        relations = g['relationships']
        relations = list(filter(lambda x: x['name'] in labels['rels'] and x['subject_id'] not in del_objs and x['object_id'] not in del_objs, relations))
        len_rel += len(relations)

        g['objects'] = objects
        g['relationships'] = relations
        n = len(objects) + len(relations)
        if n != 0:
            datap[k] = g 
    print('len(data)',len(data))
    print('len(datap)',len(datap))
    return datap


trn = json.load(open('Charades_v1_train_graph_o.json'))
tst = json.load(open('Charades_v1_test_graph_o.json'))
val = json.load(open('Charades_v1_val_graph_o.json'))
trnp = process(trn)
tstp = process(tst)
valp = process(val)
ftr = open('Charades_v1_train_graph.json','w')
fte = open('Charades_v1_test_graph.json','w')
fv = open('Charades_v1_val_graph.json','w')
json.dump(trnp,ftr)
json.dump(tstp,fte)
json.dump(valp,fv)

print('len_obj',len_obj)
print('len_att',len_att)
print('len_rel',len_rel)