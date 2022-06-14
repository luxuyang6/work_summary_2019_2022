import json
from os import F_TEST, ftruncate
data = json.load(open("Anet_val_graph.json"))
test_id = json.load(open("anet_test_id.json"))
val_id = json.load(open("anet_val_id.json"))

test_d = {}
val_d = {}
for id in data:
	vid = id[:-2]
	if vid in test_id:
		test_d[id] = data[id]
	if vid in val_id:
		val_d[id] = data[id]

ft = open("Anet_test_graph.json","w")
fv = open("Anet_val_graph.json","w")
json.dump(test_d,ft)
json.dump(val_d,fv)