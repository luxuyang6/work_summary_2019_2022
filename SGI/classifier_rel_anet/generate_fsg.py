import json
import nltk
from nltk.stem import PorterStemmer
import spacy

top_rels = 5
top_atts = 3

def convert_text(graph_input, rel_input, att_input, mode):
	out = {}
	vid_dict = {}
	for id in graph_input:
		vid = id[:-2]
		# if vid in vid_dict and mode == "test":
		# 	continue
		# else:
		if vid not in vid_dict:
			vid_dict[vid] = 1
		content = graph_input[id]
		content['objects'] = []
		content['relationships'] = []
		od = {}
		count = 0
		count_a = 0
		for i,e in enumerate(rel_input[vid]):
			if i >= top_rels:
				continue
			s,p,o = e.split("_")
			if s not in od:
				od[s] = count
				obj = {'name':s,'object_id':count,'attributes':[]}
				count += 1
				for j,a in enumerate(att_input[vid]):
					if j >= top_atts:
						continue
					if a.split("_")[0] == s and att_input[vid][a] > 0.5 :
						obj['attributes'].append(a.split("_")[1])
						count_a += 1
				content['objects'].append(obj)
			if o not in od:
				od[o] = count
				obj = {'name':o,'object_id':count,'attributes':[]}
				count += 1
				for j,a in enumerate(att_input[vid]):
					if j >= top_atts:
						continue
					if a.split("_")[0] == s and att_input[vid][a] > 0.5 :
						obj['attributes'].append(a.split("_")[1])
						count_a += 1
				content['objects'].append(obj)
			rel = {'text':e.split("_"), 'relationship_id':i+1, 'name':p, 'subject_id':od[s], 'object_id':od[o]}
			content['relationships'].append(rel)
		out[id] = content
	return out
	
	    
	

if __name__ == "__main__":
    mode = "test"
    # mode = "test"
    # mode = "train"
#     nlp = spacy.load("en_core_web_lg")

    if mode == "train":
        graph_input = json.load(open("../ActivityNet/Anet_train_graph.json"))
        rel_input = json.load(open("results/model_epoch4_train.json"))
        att_input = json.load(open("../classifier_att_anet/results/model_epoch4_train.json"))
        text_out = open('../ActivityNet/Anet_train_graph_f.json','w')
    elif mode == "test":
        graph_input = json.load(open("../ActivityNet/Anet_test_graph.json"))
        rel_input = json.load(open("results/model_epoch4_test.json"))
        att_input = json.load(open("../classifier_att_anet/results/model_epoch4_test.json"))
        text_out = open('../ActivityNet/Anet_test_graph_f.json','w')
    else:
        graph_input = json.load(open("../ActivityNet/Anet_val_graph.json"))
        rel_input = json.load(open("results/model_epoch4_val.json"))
        att_input = json.load(open("../classifier_att_anet/results/model_epoch4_val.json"))
        text_out = open('../ActivityNet/Anet_val_graph_f.json','w')

    text_dict = convert_text(graph_input, rel_input, att_input, mode)
    json.dump(text_dict, text_out)   # json.dump(text_dict, text_out)

#     if mode == "train":
#         graph_input = json.load(open("./Anet_train.json"))
#         rel_input = json.load(open("./stanford-corenlp-full-2015-12-09/anet_train.json"))
#         text_out = open('./Anet_train_graph_o.json','w')
#     else:
#         graph_input = json.load(open("./Anet_val.json"))
#         rel_input = json.load(open("./stanford-corenlp-full-2015-12-09/anet_val.json"))
#         text_out = open('./Anet_val_graph_o.json','w')
#     text_dict = convert_text(graph_input, rel_input)
#     json.dump(text_dict, text_out)   # json.dump(text_dict, text_out)