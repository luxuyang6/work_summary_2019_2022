import json
import csv
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def read_csv(filename):
    labels = {}
    action_dict = read_action_text('./Charades_v1_classes.txt')
    for key in action_dict:
        action_dict[key] = action_dict[key].lower()
        action_dict[key] = action_dict[key].replace('/',' ').split()
        #print(action_dict[key])
    with open(filename) as f:
        reader = csv.DictReader(f)
        vid = []
        keywords = [] # including objects, actions
        description = []
        for row in reader:
            _vid = row['id']
            actions = row['actions']
            if actions == '':
                #print(row)
                actions = []
                _keywords = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [action_dict[x] for x, y, z in actions]
                _keywords = []
                for action in actions:
                    _keywords = _keywords + action
            _objects = re.split(r'[/;]',row['objects'].lower())
            _keywords = _objects + _keywords
            #_description = row['descriptions'].split('.')
            _description = re.split(r'[\.;]',row['descriptions'].lower().replace(',',' ,'))
            _description = [re.split(r'[ /]',sent)+['.'] for sent in _description if sent.split() != []]
            _description = [[word for word in sent if word != ""] for sent in _description]
            vid.append(_vid)
            keywords.append(_keywords)
            description.append(_description)
        raw_data_dict = {'vid':vid, 'keywords': keywords, 'description':description}
        print('len(data)',len(raw_data_dict['keywords']))

    return raw_data_dict


def read_action_text(filename):
    # generate class_list from text file
    class_dict = {}

    ## read text file
    # f = open('./Charades_v1_classes.txt', 'r')
    f = open(filename, 'r')
    while True:
        line = f.readline()
        class_id = line.split(' ')[0]
        class_description = line[5:].strip('\n')
        class_dict[class_id] = class_description
        if not line: break
    f.close()
    return class_dict


def make_segments(data_dict, filename):
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words | {','}
    keywords = data_dict['keywords']
    #print(keywords)
    description = data_dict['description']
    data_dict['segments'] = []
    data_dict['segments_num'] = []
    ps = PorterStemmer()
    data_dict2 = []

    for i in range(len(keywords)):    
        keywords[i] = list(set(keywords[i]))
        data_dict['keywords'][i] =[w for w in keywords[i] if not w in stop_words]
        _keywords = [ps.stem(w) for w in data_dict['keywords'][i]]
        segment = []
        segment_num = []
        for j in range(len(description[i])):
            _description = [ps.stem(w) for w in description[i][j] if not w in stop_words]
            _segment = [w for w in _description if w in _keywords]            
            _segment_num = [_keywords.index(w) for w in _segment]
            _segment = [data_dict['keywords'][i][_segment_num[k]] for k in range(len(_segment))]
            segment.append(_segment)
            segment_num.append(_segment_num)
        data_dict['segments'].append(segment)
        data_dict['segments_num'].append(segment_num)
        data_i = {key:data_dict[key][i] for key in data_dict}
        data_dict2.append(data_i)


    fp = open(filename, 'w')
    json.dump(data_dict2, fp)
    fp.close()

    fp = open(filename.split('.')[0]+'2.json', 'w')
    json.dump(data_dict, fp)
    fp.close()

    return data_dict2



def get_data(filename, load_file=None):
    if load_file:
        return json.load(open(load_file))
    raw_data_dict = read_csv(filename)
    data_dict = make_segments(raw_data_dict, filename.split('.')[0]+'.json')

    return data_dict

def get_wordlist(load_file=None):
    if load_file:
        return json.load(open(load_file))
    data_dict_test = get_data('Charades_v1_test.csv','Charades_v1_test2.json')
    #data_dict_test = get_data('Charades_v1_test.csv')
    data_dict_train = get_data('Charades_v1_train.csv','Charades_v1_train2.json')
    data_dict_val = get_data('Charades_v1_val.csv','Charades_v1_val2.json')
    #data_dict_train = get_data('Charades_v1_train.csv')
    keywords = data_dict_test['keywords']+data_dict_train['keywords']+data_dict_val['keywords']
    description = data_dict_test['description']+data_dict_train['description']+data_dict_val['description']

    wordcount = {}
    keywords_list = []
    for i in range(len(description)):
        for j in range(len(keywords[i])):
            if not wordcount.get(keywords[i][j]):
                wordcount[keywords[i][j]] = 1
                keywords_list.append(keywords[i][j])

        for j in range(len(description[i])):
            for k in range(len(description[i][j])):
                if not wordcount.get(description[i][j][k]):
                    wordcount[description[i][j][k]] = 1
                else:
                    wordcount[description[i][j][k]] += 1
    
    wordlist = sorted(wordcount.items(), key=lambda x:x[1], reverse=True)              
    fp = open('wordlist.json', 'w')
    json.dump(wordlist, fp)
    fp.close()
    print("len(wordlist),wordcount[','],wordcount['a'],wordcount['pillow']: ",len(wordlist),wordcount[','],wordcount['a'],wordcount['pillow'])

    return wordlist, keywords_list



if __name__ == "__main__":
    get_data('Charades_v1_test.csv')
    get_data('Charades_v1_train.csv')
    get_data('Charades_v1_val.csv')
    get_wordlist()




    
