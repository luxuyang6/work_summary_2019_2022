from __future__ import print_function
import json
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import math as m
import csv
import pdb
np.random.seed(113)

#sys.path.append('/playpen1/home/ram/video_caption_eval')
#from automatic_evaluation import evaluate


PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[END]' # This has a vocab id, which is used at the end of untruncated target sequences





class Vocab(object):
    def __init__(self,vocab_file,max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
            Args:
                vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. 
                            This code doesn't actually use the frequencies, though.
                max_size: integer. The maximum size of the resulting Vocabulary.
                
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [PAD], [START], [STOP] and [UNK] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            wordlist = json.load(vocab_f)
            for word in wordlist:
                w = word[0]
                #if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                #    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        #print(self._word_to_id)
        #print(word)
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def print_id2word(self):
        print(self._id_to_word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def texttoidx(self,text,max_sentence_length, add_start_token=False, add_end_token=True):
        if add_end_token:
            text = text  + [STOP_DECODING]
        if add_start_token:
            text = [START_DECODING] + text
        tokens = []
        #seq_length = 0
        for word in text[:max_sentence_length]: # also need one more for [END] token
            tokens.append(self.word2id(word))


        tokens.extend([0 for i in range(max_sentence_length-len(tokens))])

        return np.asarray(tokens)

    def texttolen(self,text,max_sentence_length, add_start_token=False, add_end_token=True):
        if add_end_token:
            text = text  + [STOP_DECODING]
        if add_start_token:
            text = [START_DECODING] + text
        tokens = []
        seq_length = 0
        for word in text[:max_sentence_length]: # also need one more for [END] token
            #tokens.append(self.word2id(word))
            seq_length += 1

        #tokens.extend([0 for i in range(max_sentence_length-len(tokens))])

        return seq_length

    def segtoseg(self,text,max_sentence_length, add_start_token=False):
        text = text  + [STOP_DECODING]
        if add_start_token:
            text = [START_DECODING] + text
        tokens = []
        seq_length = 0
        for word in text[:max_sentence_length]: # also need one more for [END] token
            tokens.append(word)
            seq_length += 1

        #tokens.extend([0 for i in range(max_sentence_length-len(tokens))])

        return seq_length




class Batch(object):

    def __init__(self):
        self._dict = {}


    def put(self,key,value):
        if self._dict.get(key) is None:
            self._dict[key] = value
        else:
            raise Exception("key:{} already exits".format(key))

    def get(self,key):
       if self._dict.get(key) is not None:
           return self._dict[key]
       else:
           raise Exception("key:{} already exits".format(key))


class ChaBatcher(object):

    def __init__(self,hps,mode,vocab):
        
        self.vid_feature_path = hps.vid_feature_path
        self.max_vid_len = hps.max_vid_len
        self.max_dec_len = hps.max_dec_len
        self.mode = mode
        if self.mode=='train':
            self.batch_size = hps.batch_size
        else:
            self.batch_size = hps.test_batch_size
        self.max_text_len = hps.max_text_len
        self.vocab = vocab
        self.vid_dim = hps.vid_dim
        if mode == 'val':
            self.data = json.load(open(hps.val_file))
        elif mode == 'train':
            self.data = json.load(open(hps.train_file))
        else:
            #self.data = json.load(open(hps.val_file))
            self.data = json.load(open(hps.test_file))
            # self.data = json.load(open(hps.train_file)) + json.load(open(hps.val_file)) + json.load(open(hps.test_file))
        # self.plan_data = {}
        # reader = csv.DictReader(open(hps.test_plan_file))
        # for row in reader:
        #     self.plan_data[row['Video_ID']] = json.loads(row['Used_Plan'].replace("'","\""))
        #     #self.test_index = hps.test_index
        print("len(self.data)",len(self.data))
        self.max_enc_len = hps.max_enc_len
        self.max_group_count = hps.max_group_count
        #print(self.max_enc_len)
        #print(self.max_group_count)
        self.num_steps = m.ceil(len(self.data)/self.batch_size)

    def sort_based_on_caption_lengths(self, video_batch, video_len_batch, video_id, caption_batch, caption_len_batch, original_caption):
        sorted_indices = np.argsort(caption_len_batch)[::-1]
        return video_batch[sorted_indices], video_len_batch[sorted_indices], video_id[sorted_indices], caption_batch[sorted_indices], caption_len_batch[sorted_indices], original_caption[sorted_indices] 
    
    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test 
        Also acts as generator
        """
        if self.mode == 'train':
            np.random.shuffle(self.data)
        
        for i in range(0,len(self.data),self.batch_size):
            start = i
            if i+self.batch_size > len(self.data): # handling leftovers
                end = len(self.data)
                current_batch_size = end-start
            else:
                end = i+self.batch_size
                current_batch_size = self.batch_size
            
            batch_data = self.data[start:end]

            video_id = []
            keywords = []
            keywords_lens = []
            segments = []
            segments_gt = []
            group_count = []
            group_count_gt = []
            group_lens = []
            segments = []
            captions = []
            captions_lens = []
            texts = []
            texts_lens = []
            text_orgins = {}
            text_orgins_2 = {}

            for d in batch_data:
                video_id.append(d['vid'])
                #plan
                # if self.mode == 'test':
                #     #plan = self.plan_data[d['vid']][self.test_index]['plan']

                # pdb.set_trace()
                # plan = self.plan_data[d['vid']]
                # key_dict = {}
                # for i,keyword in enumerate(d['keywords']):
                #     key_dict[keyword] = i
                # for i,group in enumerate(plan):
                #     for j,word in enumerate(group):
                #         plan[i][j] = key_dict.get(word,0)

                keywords.append(list(self.vocab.texttoidx(d['keywords'],self.max_enc_len, add_end_token=False)))
                keywords_lens.append(self.vocab.texttolen(d['keywords'],self.max_enc_len, add_end_token=False))
                segment = []
                count1 = 0
                group_len = []
                
                for seg in d['segments_num'][:self.max_group_count-1]:
                #for seg in plan[:self.max_group_count-1]:
                    seg1 = [0]*self.max_enc_len
                    count2 = 0
                    for keyword in seg:
                        seg1[keyword] = 1
                        count2 += 1
                    group_len.append(count2)
                    count1 += 1
                    segment.append(seg1)
                #segment.append([0]*self.max_enc_len)
                segment.extend([[0]*self.max_enc_len for i in range(self.max_group_count-len(segment))])
                segments_gt.append(segment)
                group_count_gt.append(count1)

                # d['segments_num'] = plan
                segment = []
                count1 = 0
                group_len = []
                for seg in d['segments_num'][:self.max_group_count-1]:
                #for seg in plan[:self.max_group_count-1]:
                    seg1 = [0]*self.max_enc_len
                    count2 = 0
                    for keyword in seg:
                        seg1[keyword] = 1
                        count2 += 1
                    group_len.append(count2)
                    count1 += 1
                    segment.append(seg1)
                #segment.append([0]*self.max_enc_len)
                segment.extend([[0]*self.max_enc_len for i in range(self.max_group_count-len(segment))])
                segments.append(segment)
                group_count.append(count1)
                group_lens.append(group_len)
                caption = []
                captions_len = []
                text_orgin = []
                text_orgin_2 = []
                text = []
                for cap in d['description'][:self.max_group_count]:
                    caption.append(list(self.vocab.texttoidx(cap,self.max_dec_len)))
                    captions_len.append(self.vocab.texttolen(cap,self.max_dec_len))
                    text_orgin += cap
                    text_orgin_2.append(' '.join(cap))
                    text += list(self.vocab.texttoidx(cap,len(cap),add_end_token=False))
                caption.extend([[2]+[0]*(self.max_dec_len-1) for i in range(self.max_group_count-len(caption))])
                captions_len.extend([1 for i in range(self.max_group_count-len(captions_len))])
                captions.append(caption)
                captions_lens.append(captions_len)
                text = text[:self.max_text_len]
                texts_lens.append(len(text))
                text.extend([0 for i in range(self.max_text_len-len(text))])
                texts.append(text)
                text_orgins[d['vid']] = text_orgin
                text_orgins_2[d['vid']] = text_orgin_2
                        

            video_features = [pickle.load(open(os.path.join(self.vid_feature_path,key+'.npy'),'rb')) for key in video_id]
            # video_features = [np.load(os.path.join(self.vid_feature_path,key[2:]+'_resnet.npy')) for key in video_id]
            # video_feaures = []
            # for key in video_id:
            #     if not os.path.exists(os.path.join(self.vid_feature_path,key[2:]+'_resnet.npy')):
            #         video_features.append(np.zeros([50,2048]))
            #     else:
            #         video_features.append(np.load(os.path.join(self.vid_feature_path,key[2:]+'_resnet.npy')))
            #print("videofeatures",np.array(video_features).shape)
                
            # transform/clip frames
            video_batch = np.zeros((current_batch_size,self.max_vid_len,self.vid_dim))
            video_length = []
            for idx,feat in enumerate(video_features):
                if len(feat)>self.max_vid_len:
                    video_batch[idx][:] = feat[:self.max_vid_len]
                    video_length.append(self.max_vid_len)
                else:
                    video_batch[idx][:len(feat)] = feat
                    video_length.append(len(feat))

            '''if self.mode == 'train':
                video_batch, video_length, video_id, caption_batch, caption_length = self.sort_based_on_caption_lengths(
                                                                                                            np.asarray(video_batch), np.asarray(video_length), 
                                                                                                            np.asarray(video_id), np.asarray(caption_batch), 
                                                                                                            np.asarray(caption_length)) '''

            '''else:
                video_batch = np.asarray(video_batch)
                video_length = np.asarray(video_length)'''


            batch = Batch()

            keywords = torch.LongTensor(keywords)
            texts = torch.LongTensor(texts)
            segments = torch.LongTensor(segments)
            segments_gt = torch.LongTensor(segments_gt)
            captions = torch.LongTensor(captions) 
            video_batch = torch.FloatTensor(video_batch) 

            keywords = keywords[:,:max(keywords_lens)]
            texts = texts[:,:max(texts_lens)]
            segments = segments[:,:max(group_count),:max(keywords_lens)]
            segments_gt = segments_gt[:,:max(group_count_gt),:max(keywords_lens)]
            captions = captions[:,:max(group_count),:max(map(max,captions_lens))] #2维list取最大值
            video_batch = video_batch[:,:max(video_length),:]

            batch.put('keyword_batch',keywords)
            #print('longtensor of keywords',torch.LongTensor(np.asarray(keywords)))
            batch.put('keyword_len_batch',torch.tensor(keywords_lens))
            batch.put('text_batch',texts)
            batch.put('text_len_batch',torch.tensor(texts_lens))
            #print('segments_num',segments)
            batch.put('segment_batch',segments)
            batch.put('segment_gt_batch',segments_gt)
            #print('longtensor of segments_num',torch.LongTensor(segments))
            batch.put('caption_batch',captions)
            #print('longtensor of captions',torch.LongTensor(captions))
            batch.put('caption_len_batch',torch.tensor(captions_lens))
            batch.put('group_count_batch',torch.tensor(group_count))
            batch.put('group_count_gt_batch',torch.tensor(group_count_gt))
            #batch.put('group_len_batch',np.asarray(group_lens))

            batch.put('video_batch',video_batch)
            batch.put('video_len_batch',torch.tensor(video_length))
            batch.put('video_id',video_id)
            batch.put('text_orgins',text_orgins)
            batch.put('text_orgins_2',text_orgins_2)
            yield batch

class MSRVTTBatcher(object):

    def __init__(self,hps,mode,vocab):
        
        self._vid_feature_path = hps.vid_feature_path
        self._captions_path = hps.captions_path
        self._max_enc_steps = hps.encoder_rnn_max_length
        self._max_dec_steps = caption.decoder_rnn_max_length
        self._mode = mode
        self._batch_size = hps.batch_size
        self.vocab = vocab
        self._vid_dim = hps.vid_dim
        self.data,self.data_dict = self._process_data()
        self.num_steps = int(len(self.data)/self._batch_size) + 1

    def _process_data(self):
        """this module extracts data from videos and caption files and creates batches"""
        # load json data which contains all the information
        data = []
        data_dict = {}
        filename ='sents_'+self._mode+'.txt'
        with open(os.path.join(self._captions_path,filename),'r') as f:
            for line in f.read().splitlines():
                line = line.split('\t')
                vid_id = line[0]
                caption = line[1]
                data.append((vid_id,caption))
                if data_dict.get(vid_id) is None:
                    data_dict[vid_id] = [caption]
                else:
                    data_dict[vid_id].append(caption)
        
        if self._mode == 'train':
            np.random.shuffle(data)
        else:
            data,_ = zip(*data) # consider only video ids for evaluation
            data = sorted(set(data),key=data.index)
        

        return data,data_dict

    def sort_based_on_caption_lengths(self, video_batch, video_len_batch, video_id, caption_batch, caption_len_batch, original_caption):
        sorted_indices = np.argsort(caption_len_batch)[::-1]
        return video_batch[sorted_indices], video_len_batch[sorted_indices], video_id[sorted_indices], caption_batch[sorted_indices], caption_len_batch[sorted_indices], original_caption[sorted_indices] 
    
    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test 
        Also acts as generator
        """
        if self._mode == 'train':
            np.random.shuffle(self.data)
        print("len(self.data)",len(self.data))
        for i in range(0,len(self.data),self._batch_size):
            start = i
            if i+self._batch_size > len(self.data): # handling leftovers
                end = len(self.data)
                current_batch_size = end-start
            else:
                end = i+self._batch_size
                current_batch_size = self._batch_size
            if self._mode == 'train':
                video_id,original_caption = zip(*self.data[start:end])
            else:
                video_id = self.data[start:end]

            video_features = [np.load(os.path.join(self._vid_feature_path,key+'.mp4.npy')) for key in video_id]
            print("videofeatures",np.array(video_features).shape)
            if self._mode == 'train':
                caption_batch = []
                caption_length = []
                for cap in original_caption:
                    cap_id,cap_length = self.vocab.texttoidx(cap,self._max_dec_steps)
                    caption_batch.append(cap_id)
                    caption_length.append(cap_length)

            original_caption_dict = {}
            for vid in video_id:
                original_caption_dict[vid] = self.data_dict[vid]
                
            # transform/clip frames
            video_batch = np.zeros((current_batch_size,self._max_enc_steps,self._vid_dim))
            video_length = []
            for idx,feat in enumerate(video_features):
                if len(feat)>self._max_enc_steps:
                    video_batch[idx][:] = feat[:self._max_enc_steps]
                    video_length.append(self._max_enc_steps)
                else:
                    video_batch[idx][:len(feat)] = feat
                    video_length.append(len(feat))

            if self._mode == 'train':
                video_batch, video_length, video_id, caption_batch, caption_length, original_caption = self.sort_based_on_caption_lengths(
                                                                                                            np.asarray(video_batch), np.asarray(video_length), 
                                                                                                            np.asarray(video_id), np.asarray(caption_batch), 
                                                                                                            np.asarray(caption_length), np.asarray(original_caption)) 

            else:
                video_batch = np.asarray(video_batch)
                video_length = np.asarray(video_length)


            batch = Batch()
            if self._mode == 'train':
                batch.put('original_caption',original_caption)
                batch.put('caption_batch',torch.LongTensor(caption_batch))
                batch.put('caption_len_batch',caption_length)
            batch.put('original_caption_dict',original_caption_dict)
            batch.put('video_batch',torch.FloatTensor(video_batch))
            batch.put('video_len_batch',video_length)
            batch.put('video_id',video_id)
            yield batch




class SNLIBatcher(object):

    def __init__(self,max_steps,vocab):
        self._max_steps = max_steps
        self.vocab = vocab


    def process_external_data(self, prem, hypo):

        original_premise = prem
        original_hypothesis = hypo


        premise_batch = []
        premise_length = []
        hypothesis_batch = []
        hypothesis_length = []
        
        for prem, hypo in zip(original_premise, original_hypothesis):

            prem_id, prem_length = self.vocab.texttoidx(prem, self._max_steps, add_start_token=True)
            hypo_id, hypo_length = self.vocab.texttoidx(hypo, self._max_steps, add_start_token=True)
            premise_batch.append(prem_id)
            premise_length.append(prem_length)
            hypothesis_batch.append(hypo_id)
            hypothesis_length.append(hypo_length)



        batch = Batch()
        batch.put('original_premise', original_premise)
        batch.put('original_hypothesis', original_hypothesis)
        batch.put('premise_batch', torch.LongTensor(np.asarray(premise_batch)))
        batch.put('premise_length', np.asarray(premise_length))
        batch.put('hypothesis_batch', torch.LongTensor(np.asarray(hypothesis_batch)))
        batch.put('hypothesis_length', np.asarray(hypothesis_length))

        return batch

