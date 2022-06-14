import os
import sys

import argparse
import json
import pdb
import numpy as np

import caption.encoders.vanilla
import caption.decoders.vanilla
import caption.models.vanilla
import caption.models.attention

import controlimcap.encoders.gcn
import controlimcap.models.flatattn
import controlimcap.models.graphattn
import controlimcap.models.graphmemory

ENCODER = 'encoder'
DECODER = 'decoder'
MPENCODER = 'mp_encoder'
ATTNENCODER = 'attn_encoder'

ROOT_DIR = '/home/xylu/PHVM/video_asg2cap/ActivityNet/'
# ROOT_DIR = '/data1/csz/MSCOCO'

def gen_vanilla_encoder_cfg(enc_cfg, dim_fts, dim_embed):
  enc_cfg.dim_fts = dim_fts
  enc_cfg.dim_embed = dim_embed
  enc_cfg.is_embed = True
  enc_cfg.dropout = 0
  enc_cfg.norm = False
  enc_cfg.nonlinear = False
  return enc_cfg

def gen_gcn_encoder_cfg(enc_cfg, dim_input, dim_hidden):
  enc_cfg.dim_input = dim_input
  enc_cfg.dim_hidden = dim_hidden
  enc_cfg.num_rels = 6
  enc_cfg.num_bases = -1
  enc_cfg.num_hidden_layers = 2 
  enc_cfg.max_attn_len = 15
  enc_cfg.self_loop = True
  enc_cfg.num_node_types = 3
  enc_cfg.embed_first = True
  return enc_cfg

def gen_vanilla_decoder_cfg(dec_cfg, num_words, hidden_size):
  dec_cfg.rnn_type = 'lstm'
  dec_cfg.num_words = num_words
  dec_cfg.dim_word = 512
  dec_cfg.hidden_size = hidden_size
  dec_cfg.num_layers = 1
  dec_cfg.hidden2word = False
  dec_cfg.tie_embed = True
  dec_cfg.fix_word_embed = False
  dec_cfg.max_words_in_sent = 30
  dec_cfg.dropout = 0.5
  dec_cfg.schedule_sampling = False
  dec_cfg.ss_rate = 0.
  dec_cfg.ss_increase_rate = 0.05
  dec_cfg.ss_max_rate = 0.25
  dec_cfg.ss_increase_epoch = 5

  dec_cfg.greedy_or_beam = False  # test method
  dec_cfg.beam_width = 1
  dec_cfg.sent_pool_size = 1
  return dec_cfg

def gen_attn_decoder_cfg(dec_cfg, num_words, hidden_size):
  dec_cfg = gen_vanilla_decoder_cfg(dec_cfg, num_words, hidden_size)
  dec_cfg.memory_same_key_value = True
  dec_cfg.attn_input_size = 512
  dec_cfg.attn_size = 512
  dec_cfg.attn_type = 'mlp'
  return dec_cfg

def gen_common_model_cfg(model_cfg):
  model_cfg.trn_batch_size = 128
  model_cfg.tst_batch_size = 128
  model_cfg.num_epoch = 100
  model_cfg.base_lr = 2e-4
  model_cfg.monitor_iter = 1000
  model_cfg.summary_iter = 1000
  model_cfg.save_iter = -1
  model_cfg.val_iter = -1
  model_cfg.val_per_epoch = True
  model_cfg.save_per_epoch = True
  return model_cfg

def prepare_attention(mtype):

  mp_ft_name = 'Charades'
  attn_ft_name = 'node_label'

  res_dir = os.path.join(ROOT_DIR, 'results', 'VideoControlCAP')

  hidden_size = 512
  dim_attn_ft = 512
  dim_mp_ft = 1024
  num_words = len(np.load(os.path.join(ROOT_DIR, 'int2word.npy')))

  if mtype == 'node':
    model_cfg = caption.models.attention.AttnModelConfig()
  elif mtype == 'node.role':
    model_cfg = controlimcap.models.flatattn.NodeRoleBUTDAttnModelConfig()
  elif mtype in ['rgcn', 'rgcn.memory', 'rgcn.flow', 'rgcn.flow.memory']:
    model_cfg = controlimcap.models.graphattn.GraphModelConfig()

  model_cfg = gen_common_model_cfg(model_cfg)

  mp_enc_cfg = gen_vanilla_encoder_cfg(model_cfg.subcfgs[MPENCODER], [dim_mp_ft, hidden_size], hidden_size)

  if mtype in ['node', 'node.role']:
    attn_enc_cfg = gen_vanilla_encoder_cfg(model_cfg.subcfgs[ATTNENCODER], [dim_attn_ft], hidden_size)
    if mtype == 'node.role':
      attn_enc_cfg.num_node_types = 3
  elif mtype in ['rgcn', 'rgcn.memory', 'rgcn.flow', 'rgcn.flow.memory']:
    attn_enc_cfg = gen_gcn_encoder_cfg(model_cfg.subcfgs[ATTNENCODER], dim_attn_ft, hidden_size)
    attn_enc_cfg.num_node_types = 3

  dec_cfg = gen_attn_decoder_cfg(model_cfg.subcfgs[DECODER], num_words, hidden_size)
  
  output_dir = os.path.join(res_dir, mtype, 
    'mp.%s.attn.%s.%s%s.layer.%d.hidden.%d%s%s%s'%
    (mp_ft_name, attn_ft_name, 
      'rgcn.%d.'%attn_enc_cfg.num_hidden_layers if 'rgcn' in opts.mtype else '',
      dec_cfg.rnn_type,
      dec_cfg.num_layers, dec_cfg.hidden_size, 
      '.hidden2word' if dec_cfg.hidden2word else '',
      '.tie_embed' if dec_cfg.tie_embed else '',
      '.schedule_sampling' if dec_cfg.schedule_sampling else '')
    )

  if 'rgcn' in mtype:
    output_dir = '%s%s'%(output_dir, '.embed_first_mi_anet' if attn_enc_cfg.embed_first else '')
  print(output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  model_cfg.save(os.path.join(output_dir, 'model.json'))

  path_cfg = {
    'output_dir': output_dir,
    'video_dir' : '/home/shares/VideoCaption/Activitynet-Captions/training/',
    'word2int_file': os.path.join(ROOT_DIR, 'word2int.json'),
    'int2word_file': os.path.join(ROOT_DIR, 'int2word.npy'),
    'anno_file': {},
  }
  setdict = {'trn':'train','tst':'test','val':'val'}
  for setname in ['trn', 'val', 'tst']:
    path_cfg['anno_file'][setname] = os.path.join(ROOT_DIR, 'Anet_%s_graph_d.json'%setdict[setname])

  with open(os.path.join(output_dir, 'path.json'), 'w') as f:
    json.dump(path_cfg, f, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mtype', 
    choices=['node', 'node.role', 'rgcn', 'rgcn.flow', 'rgcn.memory', 'rgcn.flow.memory'])
  opts = parser.parse_args()

  prepare_attention(opts.mtype)
 
