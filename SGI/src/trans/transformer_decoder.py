import torch
import torch.nn as nn
from trans.common import *
import time
import pdb
import torch.nn.functional as F


class Embedder(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embed = nn.Embedding(vocab_size, d_model)
  def forward(self, x):
    return self.embed(x)


class DecoderLayer(nn.Module):
  def __init__(self, d_model, heads, dropout=0.1):
    super().__init__()
    self.norm_1 = Norm(d_model)
    self.norm_2 = Norm(d_model)
    self.norm_3 = Norm(d_model)
    self.norm_4 = Norm(d_model)
    self.norm_5 = Norm(d_model)
    self.norm_6 = Norm(d_model)
    self.norm_7 = Norm(d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)
    self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.ff = FeedForward(d_model, dropout=dropout)
    self.address_layer = nn.Sequential(
      nn.Linear(d_model*2, d_model),
      nn.ReLU(),
      nn.Linear(d_model, 3))
    self.address_layer2 = nn.Sequential(
      nn.Linear(d_model*2, d_model),
      nn.ReLU(),
      nn.Linear(d_model, 4))

  def forward(self, x, src_1, src_11, src_mask_1, src_2, src_22, src_mask_2, trg_mask, flow_edges, layer_cache=None):
    # pdb.set_trace()
    x2 = self.norm_1(x)
    x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask, layer_cache=layer_cache, attn_type='self')[0])
    x2 = self.norm_2(x)
    src_1 = self.norm_4(src_1)
    src_2 = self.norm_5(src_2)
    src_11 = self.norm_6(src_11)
    src_22 = self.norm_7(src_22)
    context_1, attn_1 = self.attn_2(x2, src_1, src_11, src_mask_1, attn_type='context')
    context_2, attn_2 = self.attn_3(x2, src_2, src_22, src_mask_2, attn_type='context')

    address_params = self.address_layer(torch.cat([context_1, context_2], -1))
    interpolate_gate = torch.softmax(address_params, dim=-1)

    address_params2 = self.address_layer2(torch.cat([context_1[:,0,:], context_2[:,0,:]], -1))
    flow_gate = torch.softmax(address_params2, dim=-1)

    batch_size = src_2.size(0)
    max_attn_len = src_2.size(1)
    device = x.device

    init_attn = torch.zeros((batch_size, max_attn_len)).to(device) 
    init_attn[:, 0] = 1
    
    flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, init_attn)
    flow_attn_score_2 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_1)
    flow_attn_score_3 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_2)
    # (batch, max_attn_len, 3)
    flow = torch.stack([x.view(batch_size, max_attn_len) \
      for x in [init_attn, flow_attn_score_1, flow_attn_score_2, flow_attn_score_3]], 2)
    flow_attn_score = torch.sum(flow_gate.unsqueeze(1) * flow, 2)
    # pdb.set_trace()
    prevent_attn = flow_attn_score
    attn_3 = flow_attn_score.unsqueeze(1)

    for i in range(1, x.size(1)):
      address_params2 = self.address_layer2(torch.cat([context_1[:,i,:], context_2[:,i,:]], 1))
      flow_gate = torch.softmax(address_params2, dim=-1)
      
      flow_attn_score_1 = torch.einsum('bts,bs->bt', flow_edges, prevent_attn)
      flow_attn_score_2 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_1)
      flow_attn_score_3 = torch.einsum('bts,bs->bt', flow_edges, flow_attn_score_2)
      # (batch, max_attn_len, 3)
      flow = torch.stack([x.view(batch_size, max_attn_len) \
        for x in [prevent_attn, flow_attn_score_1, flow_attn_score_2, flow_attn_score_3]], 2)
      flow_attn_score = torch.sum(flow_gate.unsqueeze(1) * flow, 2)
      prevent_attn = flow_attn_score
      attn_3 = torch.cat([attn_3, flow_attn_score.unsqueeze(1)], 1)
    src_mask_2 = src_mask_2.unsqueeze(1)
    # pdb.set_trace()
    attn_3 = attn_3.masked_fill(src_mask_2.squeeze(1) == 0, -1e9) 
    attn_3 = F.softmax(attn_3, dim=-1) 
    context_3 = torch.matmul(attn_3, src_22)

    # pdb.set_trace()
    context = interpolate_gate[:,:,0].unsqueeze(-1)*context_1 + interpolate_gate[:,:,1].unsqueeze(-1)*context_2 + interpolate_gate[:,:,2].unsqueeze(-1)*context_3
    x = x + self.dropout_2(context)
    x2 = self.norm_3(x)
    x = x + self.dropout_3(self.ff(x2))
    return x, attn_2
    
    
class Decoder(nn.Module):
  def __init__(self, vocab_size, d_model, N, heads, dropout):
    super().__init__()
    self.N = N
    self.embed = Embedder(vocab_size, d_model)
    self.pe = PositionalEncoder(d_model, dropout=dropout)
    self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
    self.norm = Norm(d_model)
    self.cache = None
  
  def _init_cache(self):
    self.cache = {}
    for i in range(self.N):
      self.cache['layer_%d'%i] = {
        'self_keys': None,
        'self_values': None,
      }    

  def forward(self, trg, src_1, src_11, src_mask_1, src_2, src_22, src_mask_2,  trg_mask, flow_edges, step=None):
    if step == 1:
      self._init_cache()

    # pdb.set_trace()

    x = self.embed(trg)
    x = self.pe(x, step)
    attn_w = []
    for i in range(self.N):
      layer_cache = self.cache['layer_%d'%i] if step is not None else None
      x, attn = self.layers[i](x, src_1, src_11, src_mask_1, src_2, src_22, src_mask_2, trg_mask, flow_edges, layer_cache=layer_cache)
      attn_w.append(attn)
    return self.norm(x), sum(attn_w)/self.N



