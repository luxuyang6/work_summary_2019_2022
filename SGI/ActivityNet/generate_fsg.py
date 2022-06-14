import json
import nltk
from nltk.stem import PorterStemmer
import spacy



def convert_text(graph_input, cla_input):
	out = {}
	vid_dict = {}
	for id in graph_input:
		vid = id[:-2]
		if vid in vid_dict:
			continue
		else:
			vid_dict[vid] = 1
		content = graph_input[id]
		content['objects'] = []
		content['relationships'] = []
		od = {}
		count = 0
		for i,e in enumerate(cla_input[vid]):
			if i >= 5:
				continue
			if e[0] not in od:
				od[e[0]] = count
				obj = {'name':e[0],'object_id':count,'attributes':[]}
				count += 1
				content['objects'].append(obj)
			if e[2] not in od:
				od[e[2]] = count
				obj = {'name':e[2],'object_id':count,'attributes':[]}
				count += 1
				content['objects'].append(obj)
			rel = {'text':e, 'relationship_id':i+1, 'name':e[1], 'subject_id':od[e[0]], 'object_id':od[e[2]]}
			content['relationships'].append(rel)
		out[id] = content
	return out
	
	    
	

if __name__ == "__main__":
  
	cls_input = json.load(open("anet_out_dict_raw.json"))
	graph_input = json.load(open("Anet_train_graph.json"))
	text_out = open('Anet_train_graph_f.json','w')
	text_dict = convert_text(graph_input, cls_input)
	json.dump(text_dict, text_out)   # json.dump(text_dict, text_out)

	graph_input = json.load(open("Anet_test_graph.json"))
	text_out = open('Anet_test_graph_f.json','w')
	text_dict = convert_text(graph_input, cls_input)
	json.dump(text_dict, text_out)   # json.dump(text_dict, text_out)

	graph_input = json.load(open("Anet_val_graph.json"))
	text_out = open('Anet_val_graph_f.json','w')
	text_dict = convert_text(graph_input, cls_input)
	json.dump(text_dict, text_out)   # json.dump(text_dict, text_out)

#     if mode == "train":
#         graph_input = json.load(open("./Anet_train.json"))
#         cls_input = json.load(open("./stanford-corenlp-full-2015-12-09/anet_train.json"))
#         text_out = open('./Anet_train_graph_o.json','w')
#     else:
#         graph_input = json.load(open("./Anet_val.json"))
#         cls_input = json.load(open("./stanford-corenlp-full-2015-12-09/anet_val.json"))
#         text_out = open('./Anet_val_graph_o.json','w')
#     text_dict = convert_text(graph_input, cls_input)
#     json.dump(text_dict, text_out)   # json.dump(text_dict, text_out)