import json



def process(data):
    datap = {}
    for k, g in data.items():
        k0 = k.split('_')[0]
        if datap.get(k0)==None:
            datap[k0] = {}
            datap[k0]['vid'] = k0
            datap[k0]['keywords'] = []
            datap[k0]['description'] = []
            datap[k0]['segments'] = []
            datap[k0]['segments_num'] = []
            num = 0
            keyword_dict = {}

            datap[k0]['description'].append(g['phrase'].split())
            segment = []
            segment_num = []
            for o in g['objects']:
                # if o['name'] == 'person':
                #     continue
                if keyword_dict.get(o['name'])==None:
                    datap[k0]['keywords'].append(o['name']) 
                    keyword_dict[o['name']] = num
                    num += 1
                segment.append(o['name'])
            #     segment_num.append(keyword_dict[o['name']])
            #     for a in o['attributes']:
            #         if keyword_dict.get(a)==None:
            #             datap[k0]['keywords'].append(a) 
            #             keyword_dict[a] = num
            #             num += 1
            #         segment.append(a)
            #         segment_num.append(keyword_dict[a])
            # for r in g['relationships']:
            #     if keyword_dict.get(r['name'])==None:
            #         datap[k0]['keywords'].append(r['name']) 
            #         keyword_dict[r['name']] = num
            #         num += 1
            #     segment.append(r['name'])
            #     segment_num.append(keyword_dict[r['name']])
            datap[k0]['segments'].append(segment)
            datap[k0]['segments_num'].append(segment_num)

        else:
            datap[k0]['description'].append(g['phrase'].split())
            segment = []
            segment_num = []
            for o in g['objects']:
                # if o['name'] == 'person':
                #     continue
                if keyword_dict.get(o['name'])==None:
                    datap[k0]['keywords'].append(o['name']) 
                    keyword_dict[o['name']] = num
                    num += 1
                segment.append(o['name'])
                segment_num.append(keyword_dict[o['name']])
            #     for a in o['attributes']:
            #         if keyword_dict.get(a)==None:
            #             datap[k0]['keywords'].append(a) 
            #             keyword_dict[a] = num
            #             num += 1
            #         segment.append(a)
            #         segment_num.append(keyword_dict[a])
            # for r in g['relationships']:
            #     if keyword_dict.get(r['name'])==None:
            #         datap[k0]['keywords'].append(r['name']) 
            #         keyword_dict[r['name']] = num
            #         num += 1
            #     segment.append(r['name'])
            #     segment_num.append(keyword_dict[r['name']])
            datap[k0]['segments'].append(segment)
            datap[k0]['segments_num'].append(segment_num)

    print('len(data)',len(data))
    print('len(datap)',len(datap))
    return list(datap.values())

# captions = json.load(open("charades_captions.txt"))

# trn = json.load(open('Charades_v1_train_graph_b.json'))
tst = json.load(open('Charades_v1_test_graph_d.json'))
# val = json.load(open('Charades_v1_val_graph_b.json'))
# trnp = process(trn)
tstp = process(tst)
# valp = process(val)
# ftr = open('Charades_v1_train_obj_f.json','w')
fte = open('Charades_v1_test_obj_d.json','w')
# fv = open('Charades_v1_val_obj_f.json','w')
# json.dump(trnp,ftr)
json.dump(tstp,fte)
# json.dump(valp,fv)
