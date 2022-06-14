import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb

import framework.configbase
import framework.ops
from framework.modules.embeddings import Embedding


'''
Vanilla Encoder: embed nd array (batch_size, ..., dim_ft)
  - EncoderConfig
  - Encoder

Multilayer Perceptrons: feed forward networks + softmax
  - MLPConfig
  - MLP
'''

class EncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = [2048]
    self.dim_embed = 512
    self.is_embed = True
    self.dropout = 0
    self.norm = False
    self.nonlinear = False

  def _assert(self):
    if not self.is_embed:
      assert self.dim_embed == sum(self.dim_fts)

class Encoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    if self.config.is_embed:
      self.ft_embed = nn.Linear(sum(self.config.dim_fts), self.config.dim_embed)
    self.dropout = nn.Dropout(self.config.dropout)

  def forward(self, fts):
    '''
    Args:
      fts: size=(batch, ..., sum(dim_fts))
    Returns:
      embeds: size=(batch, dim_embed)
    '''
    embeds = fts
    if self.config.is_embed:
      embeds = self.ft_embed(embeds)
    if self.config.nonlinear:
      embeds = F.relu(embeds)
    if self.config.norm:
      embeds = framework.ops.l2norm(embeds) 
    embeds = self.dropout(embeds)
    return embeds




class FramesEncoder(nn.Module):
    def __init__(self):
        super(FramesEncoder, self).__init__()

        # self.config = args
        self.vid_dim = 2048
        # self.dim_fts = 1024
        self.embed_size = 512
        self.hidden_dim = 512
        self.max_video_len = 100
        self.enable_cuda = True
        self.layers = 1
        self.birnn = True
        self.dropout_rate = 0.5

        self.linear = nn.Linear(self.vid_dim, self.embed_size, bias=False)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_dim, self.layers, batch_first=True, bidirectional=self.birnn, dropout=self.dropout_rate)
        self.video_linear1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.video_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        

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

        flengths, idx_sort = np.sort(flengths.cpu().numpy())[::-1], np.argsort(-flengths.cpu().numpy())
        if self.enable_cuda:
            frames = frames.index_select(0, torch.cuda.LongTensor(idx_sort))
        else:
            frames = frames.index_select(0, torch.LongTensor(idx_sort))



        frames = self.linear(frames)
        #frame_packed = nn.utils.rnn.pack_padded_sequence(frames, flengths, batch_first=True)
        frame_packed = nn.utils.rnn.pack_padded_sequence(frames, flengths.copy(), batch_first=True)
        outputs, (ht, ct) = self.rnn(frame_packed, self.init_rnn)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True, total_length=self.max_video_len)

        idx_unsort = np.argsort(idx_sort)
        if self.enable_cuda:
            outputs = outputs.index_select(0, torch.cuda.LongTensor(idx_unsort))
        else:
            outputs = outputs.index_select(0, torch.LongTensor(idx_unsort))

        outputs_t = torch.zeros(outputs.shape[0],outputs.shape[2]).cuda()
        for i in range(batch_size):
            outputs_t[i] = outputs[i,flengths[i]-1,:]  # get last state

        outputs = self.video_linear1(outputs)

        return outputs, outputs_t



class EmbedLayer(nn.Module):
    def __init__(self, args):
        super(EmbedLayer, self).__init__()

        self.video_dim = 2048
        self.hidden_dim = 512
        self.config = args

        self.embedding = Embedding(self.config.num_words,
          self.config.dim_word, fix_word_embed=self.config.fix_word_embed)

        # self.linear = nn.Linear(self.video_dim, self.config.dim_word)
        self.graph_linear = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, wordids, bow=False):
        # pdb.set_trace()
        if not bow:
        # single word nodes
          wordemb = self.embedding(wordids)
        else:
        # bow of multiple words in a node
          emb = self.embedding(wordids)
          wordsum = torch.sum(wordids.eq(0).eq(0), -1, keepdim=True)
          wordsum = wordsum | wordsum.eq(0).long()
          wordemb = torch.sum(emb, -2)/wordsum
        
        # return self.linear(torch.cat([wordemb, reg_fts],2))
        return wordemb
        # return wordemb + self.linear(reg_fts)
        

        
        
