import json
import numpy as np

trn = json.load(open('Anet_train_graph.json'))
tst = json.load(open('Anet_test_graph.json'))
val = json.load(open('Anet_val_graph.json'))

data = trn.copy()
data.update(tst)
data.update(val)
word_dict = {}
word2int = {}
for i, g in enumerate(data.values()):
    sent = g['phrase'].split()
    name = []
    for o in g['objects']:
        sent += o['name'].split()
        sent += o['attributes']
    for r in g['relationships']:
        sent += r['name'].split()
    for word in sent:
        if word_dict.get(word) == None:
            word_dict[word] = 0
        else:
            word_dict[word] += 1
    if i % 5000 ==0: 
        print(i)
print(sent)
sorted_ = sorted(word_dict.items(), key = lambda kv:(-kv[1], kv[0]))
int2word = [t[0] for t in sorted_]
int2word = int2word[:10000]
int2word.insert(0, '<UNK>')
int2word.insert(0, '<EOS>')
int2word.insert(0, '<BOS>')

for i, word in enumerate(int2word):
    word2int[word] = i

f_w2i = open('word2int.json','w')
json.dump(word2int,f_w2i)
np.save("int2word.npy",np.array(int2word, dtype='<U17'))

print(len(int2word))
print(len(word2int))
print('finished')




