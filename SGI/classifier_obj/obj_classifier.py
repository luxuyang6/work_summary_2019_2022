# -*- coding:utf8 -*-#
import os
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import numpy as np
import pdb
from numpy import *
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='test',
                        choices=["test","train"])
parser.add_argument("--dataset", default='test',
                        choices=["test","train","val"])
parser.add_argument("--load_path", default='./model/model_epoch9.pth')

args = parser.parse_args()
# config
mode = args.mode
load_path = args.load_path
dataset = args.dataset
EPOCH = 10
augment_rate = 10
# keyfrmae
# keyframe = 10
max_video_len = 100
BATCH_SIZE = 256
EMBED_DIM = 300
feature_dim = 512
mpu_dim = 512
save_fr = 1
topk = 20
# KERNEL_SIZES = [3,4,5]
# KERNEL_DIM = 100
LR = 0.0001

vocab_path = './vocab.json'
data_path = '/home/xylu/PHVM/video_asg2cap/Charades/'
# video_path = '/home/xylu/video_captioning/PHVM/charades-algorithms/Charades_feature/'
video_path = '/home/xylu/PHVM/PHVM_lxy/data/Charades_feature'
# data process
if mode == 'train':
    train_data_o = json.load(open(data_path+'Charades_object_classifier_train_20.json'))
    data = []
    for vid in train_data_o:
        for obj in train_data_o[vid]:
            if train_data_o[vid][obj] == 1:
                for i in range(augment_rate):
                    data.append((([obj], vid), train_data_o[vid][obj]))
            else:
                data.append((([obj], vid), train_data_o[vid][obj]))
    # data = [(([obj], vid), train_data_o[vid][obj]) for vid in train_data_o for obj in train_data_o[vid]]
else:
    test_data_o = json.load(open(data_path+'Charades_object_classifier_' +dataset+'_20.json'))
    data = [(([obj], vid), test_data_o[vid][obj]) for vid in test_data_o for obj in test_data_o[vid]]

flatten = lambda l: [item for sublist in l for item in sublist]

# pdb.set_trace()

inp, Label = list(zip(*data))
inp = list(inp)
Label = list(Label)
X,Y = list(zip(*inp))
X = list(X)
Y = list(Y)
vids = set(Y)
video_fts = {}
for vid in vids:
    video_fts[vid] = pickle.load(open(os.path.join(video_path,vid+'.npy'),'rb'))

data_p = list(zip(X, Y, Label))
if mode == 'train':
    random.shuffle(data_p)


if os.path.exists(vocab_path):
    vocab = json.load(open(vocab_path))
else:
    vocab = list(set(flatten(X)))
    f_vocab = open(vocab_path,'w')
    json.dump(vocab, f_vocab)
# print(len(vocab))
word2index={'<PAD>': 0, '<UNK>': 1}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)
index2word = {v:k for k, v in word2index.items()}
print(len(word2index))
print(set(Label))
# print(len(word2index))
# print(set(Label))
# target2index = {}
# for cl in set(label):
#     if target2index.get(cl) is None:
#         target2index[cl] = len(target2index)
# index2target = {v:k for k, v in target2index.items()}
# 处理成input数组
# 先试着不用region
# 先处理vocab，将两个数据集的labels合并
    # vocab
    # video features
    # region features
# batcher
# trainer 
# model
    # rnn
    # mpu
    # linear
    # binary entropy loss

# config and settings
# flatten = lambda l: [item for item in l]
random.seed(1024)
USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def getBatch(batch_size, train_data):
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def pad_to_batch(batch):
    # x,y,label = zip(*batch)
    # pdb.set_trace()
    max_x = 1
    max_y = max_video_len
    x_p = []
    y_p = []
    x, y, vid_len, label = [], [], [], []
    for i,pair in enumerate(batch):
        xi = prepare_sequence(pair[0], word2index)
        yi = video_fts[pair[1]]

        vid_len.append(Variable(LongTensor([min(video_fts[pair[1]].shape[0], max_y)])).view(1,-1))
        label.append(Variable(LongTensor([pair[2]])).view(1, -1))

        if len(xi) < max_x:
            x_p.append(Variable(LongTensor(xi + word2index['<PAD>'] * (max_x - len(xi)))).view(1, -1))
        else:
            x_p.append(Variable(LongTensor(xi[:max_x])).view(1, -1))
        if len(yi) < max_y:
            tem_y = np.zeros([max_y, yi.shape[1]])
            tem_y[:len(yi),:] = yi
            y_p.append(Variable(FloatTensor(tem_y)).unsqueeze(0))
        else:
            y_p.append(Variable(FloatTensor(yi[:max_y])).unsqueeze(0))
        
        # x.append(prepare_sequence(pair[0], word2index).view(1, -1))
        # y.append(Variable(FloatTensor(video_fts[pair[1]])).unsqueeze(0))
        # vid_len.append(Variable(LongTensor([max(video_fts[pair[1]].shape[0], max_y)])).view(1,-1))
        # label.append(Variable(LongTensor([target2index[pair[2]]])).view(1, -1))

        # if x[i].size(1) < max_x:
        #     x_p.append(torch.cat([x[i], Variable(LongTensor([word2index['<PAD>']] * (max_x - x[i].size(1)))).view(1, -1)], 1))
        # else:
        #     x_p.append(x[i][:,:max_x])
        # if y[i].size(1) < max_y:
        #     y_p.append(torch.cat([y[i], Variable(FloatTensor(np.zeros([1, max_y - y[i].size(1), y[i].size(2)])))], 1))
        # else:
        #     y_p.append(y[i][:,:max_y])
    # pdb.set_trace()
    out1 = torch.cat(x_p)
    out2 = torch.cat(y_p)
    out3 = torch.cat(vid_len).view(-1)
    out4 = torch.cat(label).view(-1)
    return out1, out2, out3, out4

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return idxs
    # return Variable(LongTensor(idxs))

class FramesEncoder(nn.Module):
    def __init__(self):
        super(FramesEncoder, self).__init__()

        # self.config = args
        self.vid_dim = 2048
        self.dim_fts = 1024
        self.embed_size = 512
        self.hidden_dim = 512
        self.max_video_len = 100
        self.enable_cuda = True
        self.layers = 1
        self.birnn = True
        # self.dropout_rate = 0.5

        self.linear = nn.Linear(self.vid_dim, self.embed_size, bias=False)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_dim, self.layers, batch_first=True, bidirectional=self.birnn)
        

    def init_hidden(self, batch_size):
        if self.birnn:
            return (torch.zeros(2*self.layers, batch_size, self.hidden_dim),
                    torch.zeros(2*self.layers, batch_size, self.hidden_dim))



    def forward(self, frames, flengths):
        """Handles variable size frames
           frame_embed: video features
           flengths: frame lengths
        """
        batch_size = frames.shape[0]
        #frames = self.linear(frames)
        #frames = self.dropout(frames) # adding dropout layer
        self.init_rnn = self.init_hidden(batch_size)
        if self.enable_cuda:
            self.init_rnn = self.init_rnn[0].cuda(), self.init_rnn[1].cuda()

        frames = self.linear(frames)
        if batch_size > 1:

            flengths_f, idx_sort = np.sort(flengths.cpu().numpy())[::-1], np.argsort(-flengths.cpu().numpy())
            if self.enable_cuda:
                frames = frames.index_select(0, torch.cuda.LongTensor(idx_sort))
            else:
                frames = frames.index_select(0, torch.LongTensor(idx_sort))



            #frame_packed = nn.utils.rnn.pack_padded_sequence(frames, flengths, batch_first=True)
            frames_packed = nn.utils.rnn.pack_padded_sequence(frames, flengths_f.copy(), batch_first=True)
        outputs, (ht, ct) = self.rnn(frames_packed, self.init_rnn)
        if batch_size > 1:
            outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)

            idx_unsort = np.argsort(idx_sort)
            if self.enable_cuda:
                outputs = outputs.index_select(0, torch.cuda.LongTensor(idx_unsort))
            else:
                outputs = outputs.index_select(0, torch.LongTensor(idx_unsort))

        outputs_t = torch.zeros(outputs.shape[0],outputs.shape[2]).cuda()
        for i in range(batch_size):
            outputs_t[i] = outputs[i,flengths[i]-1,:]  # get last state
            # outputs_t[i] = outputs[i,-1,:]  # get last state
        # pdb.set_trace()
        return outputs_t



class  MPUClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, feature_dim, mpu_dim, output_size, dropout=0.5):
    
    # def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super(MPUClassifier,self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.EmbedLinear = nn.Linear(embed_dim, mpu_dim) 
        self.FeatureLinear = nn.Linear(feature_dim*2, mpu_dim) 
        self.MLPLinear1 = nn.Linear(mpu_dim*4, mpu_dim) 
        self.MLPLinear2 = nn.Linear(mpu_dim, output_size) 
        self.visual_encoder = FramesEncoder()
        # self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])

        # kernal_size = (K,D) 
        self.dropout = nn.Dropout(dropout)
    
    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False


    def forward(self, x, feature, vid_len, is_training=False):
        
        x = F.normalize(torch.sum(self.EmbedLinear(self.embed(x)), 1))
        # batch * mpu_dim
        y = self.visual_encoder(feature, vid_len)
        y = F.normalize(self.FeatureLinear(y))
        # batch * mpu_dim
        z = torch.cat([torch.mul(x,y), x+y, torch.cat([x,y],1)], 1)
        if is_training:
            z = self.dropout(z)
        out = self.MLPLinear2(self.MLPLinear1(z))
        # if is_training:
        #     concated = self.dropout(concated) # (N,len(Ks)*Co)
        # out = self.fc(concated) 
        return out

model = MPUClassifier(len(word2index), EMBED_DIM, feature_dim, mpu_dim, len(set(Label)))
#model.init_weights(pretrained_vectors) # initialize embedding matrix using pretrained vectors
if USE_CUDA:
    model = model.cuda()
if mode == 'train':
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCH):
        losses = []
        for i,batch in enumerate(getBatch(BATCH_SIZE, data_p)):
            # pdb.set_trace()

            inputs, videos, vid_lens, targets = pad_to_batch(batch)
            model.zero_grad()
            preds = model(inputs, videos, vid_lens, True)
            # print(preds,targets)
            loss = loss_function(preds, targets)
            losses.append(loss.data.tolist())
            loss.backward()
            #for param in model.parameters():
            #    param.grad.data.clamp_(-3, 3)
            optimizer.step()
            
            if i % 100 == 0:
                print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
                # print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
                losses = []
        # pdb.set_trace()
        if epoch % save_fr ==0:
            path = './model/model_epoch{}.pth'.format(epoch)
            torch.save(model.state_dict(),path)

else:
    model.load_state_dict(torch.load(load_path))
    print('Loaded:',load_path)
    # print('Loaded:',load_path)

    result = []
    result_dict = {}
    pred_dict = {}
    for i,batch in enumerate(getBatch(BATCH_SIZE, data_p)):

    # for i, test in enumerate(data_p):
        if i % 100 ==0:
            print(i)
        # if i>1000:
        #     break
        inputs, videos, vid_lens, _ = pad_to_batch(batch)
        # inputs = Variable(LongTensor(prepare_sequence(test[0], word2index))).view(1, -1)
        # videos = Variable(FloatTensor(video_fts[test[1]])).unsqueeze(0)
        # videos = Variable(FloatTensor(video_fts[test[1]][keyframe])).unsqueeze(0).unsqueeze(0)
        # vid_lens = Variable(LongTensor([min(video_fts[test[1]].shape[0], max_video_len)])).view(1,-1)
        # pdb.set_trace()
        out = F.softmax(model(inputs, videos, vid_lens), -1)
        for j, test in enumerate(batch):
            vid = test[1]
            if result_dict.get(vid) == None:       
                pred_dict[vid] = []     
                result_dict[vid] = []
            result_dict[vid].append((''.join(test[0]), out[j,1].data.tolist()))
            if test[2] == 1:
                pred_dict[vid].append(''.join(test[0]))
        # if eval_mode == '1':
        #     pred = out.max(1)[1]
        #     pred = pred.data.tolist()[0]
        #     target = test[2]
        #     if pred == target:
        #         accuracy += 1
        #         if pred == 1:
        #             result.append(test_data_l[i][0])
    out_dict = {}

    recs = []
    recs_10 = []
    recs_5 = []
    accs = []

    for vid in result_dict:
        out_dict[vid] = {}
        objs, scores = list(zip(*result_dict[vid]))
        objs = list(objs)
        scores = list(scores)
        sort_ind = np.argsort(-np.array(scores)).tolist()
        acc = 0
        acc_5 = 0
        acc_10 = 0
        for j in range(topk):
            if j >= len(scores):
                continue
            out_dict[vid][objs[sort_ind[j]]] = scores[sort_ind[j]]
            if objs[sort_ind[j]] in pred_dict[vid]:
                acc +=1
                if j < 10:
                    acc_10 += 1
                    if j < 5:
                        acc_5 += 1
        # accs.append(acc/topk)
        if len(pred_dict[vid]) != 0:
            recs.append(acc/len(pred_dict[vid]))
            recs_10.append(acc_10/len(pred_dict[vid]))
            recs_5.append(acc_5/len(pred_dict[vid]))

    precesion = {}
    # precesion['acc'] = mean(accs) * 100
    precesion['rec@{}'.format(topk)] = mean(recs) * 100
    precesion['rec@5'] = mean(recs_5) * 100
    precesion['rec@10'] = mean(recs_10) * 100
    print(precesion)

    # print(pred_dict)
    f_out = open('./results/model_epoch{}_{}.json'.format(load_path[-5:-4], dataset), 'w')
    # f_out = open('./results/model_epoch{}_{}_keyframe_{}.json'.format(load_path[-5:-4], dataset, keyframe), 'w')
    json.dump(out_dict, f_out)

    result_out = open('./results/model_epoch{}_{}_result.json'.format(load_path[-5:-4], dataset), 'w')
    # result_out = open('./results/model_epoch{}_{}_result_keyframe_{}.json'.format(load_path[-5:-4], dataset, keyframe), 'w')
    json.dump(precesion, result_out)
