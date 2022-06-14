import json
from automatic_evaluation import evaluate

res_dict = json.load(open('../epoch.55.json'))
gts_list = json.load(open('../data/preprocess/Charades_v1_test.json'))
gts_dict = {}
for d in gts_list:
    gts_dict[d['vid']] = d['description']
res = {}
gts = {}

for i,key in enumerate(res_dict.keys()):
    res[i] = res_dict[key]
    vid = key.split('_')[0]
    gts[i] = [' '.join(sent) for sent in gts_dict[vid]]


print(len(res))
print(res[i])
print(gts[i])
scores = evaluate(gts, res, score_type='macro', tokenized=True)
scores_dict = {}
print("Results:")
# logger.info("Results:")
for method, score in scores:
    scores_dict[method] = score
    print("{}:{}".format(method,score))
    # logger.info(f"{method}:{score}")
f = open('../epoch.55.result.json','w')
json.dump(scores_dict,f)
