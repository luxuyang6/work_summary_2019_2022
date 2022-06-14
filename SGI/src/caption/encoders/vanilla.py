import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb

import framework.configbase
import framework.ops
from framework.modules.embeddings import Embedding
from trans.transformer_encoder import Encoder as trans_encoder
from torch.autograd import Variable


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
        self.hidden_dim = 512
        self.max_video_len = 100
        self.enable_cuda = True
        self.layers = 3
        self.heads = 8
        self.dropout_rate = 0.2
        self.trans_encoder = trans_encoder(self.vid_dim, self.hidden_dim, self.layers, self.heads, self.dropout_rate)
        self.video_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        

    def forward(self, frames, flengths):
        """Handles variable size frames
           frame_embed: video features
           flengths: frame lengths
        """
        src_mask = self.attn_mask(flengths, max_len=frames.size(1)).unsqueeze(-2)
        outputs, org_key, select = self.trans_encoder(frames, src_mask)
        return outputs

    def nopeak_mask(self, size):
        np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
        return np_mask

    def create_masks(self, ft_len, max_len, trg):
        if trg is not None:
          trg_mask = (trg != 1).unsqueeze(-2)
          size = trg.size(1) # get seq_len for matrix
          np_mask = self.nopeak_mask(size)
          trg_mask = trg_mask & np_mask  
        else:
          trg_mask = None
        src_mask = self.attn_mask(ft_len, max_len=max_len).unsqueeze(-2)
        return src_mask, trg_mask

    def attn_mask(self, lengths, max_len=None):
        ''' Creates a boolean mask from sequence lengths.
            lengths: LongTensor, (batch, )
        '''
        batch_size = lengths.size(0)
        max_len = max_len or lengths.max()
        return ~(torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .ge(lengths.unsqueeze(1)))



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
        

        
        
