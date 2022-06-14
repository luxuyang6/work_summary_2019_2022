import os
import json
import h5py
import numpy as np
from scipy import sparse
import collections
import math
import pickle
import torch
from glob import glob
import json
import pdb

import caption.readers.base

NUM_RELS = 6
UNK_WORDEMBED = np.zeros((300, ), dtype=np.float32)
PIXEL_REDUCE = 1

class ImageSceneGraphFlatReader(caption.readers.base.CaptionDatasetBase):
  # def __init__(self, name_file, mp_ft_file, obj_ft_dir, region_anno_dir, 
  #   word2int_file, max_attn_len=10, max_words_in_sent=15, 
  #   is_train=False, return_label=False, _logger=None,
  #   pred_caption_file=None):
  
  def __init__(self, anno_file, vid_dir, word2int_file, max_attn_len=10, 
    max_video_len=100, max_words_in_sent=15, is_train=False, max_name_len=3,
    return_label=False, _logger=None):

    super().__init__(word2int_file, max_words_in_sent=max_words_in_sent,
      is_train=is_train, return_label=return_label, _logger=_logger)

    self.max_attn_len = max_attn_len
    self.max_name_len = max_name_len
    self.max_words_in_sent = max_words_in_sent
    self.max_video_len = max_video_len
    self.anno_data = json.load(open(anno_file))
    self.data_ids = list(self.anno_data.keys())
    self.vids = [data_id[2:-2] for data_id in self.data_ids]
    # self.vids = [data_id.split('_')[0] for data_id in self.data_ids]
    self.obj_ids = []
    self.vid_dir = vid_dir
    # self.video_fts = [pickle.load(open(os.path.join(vid_dir+'Charades_feature/',key+'.npy'),'rb')) for key in self.vids]
    # self.video_fts = [np.load(os.path.join(vid_dir+key+'_resnet.npy')) for key in self.vids]
    # self.video_fts = [pickle.load(open(os.path.join(vid_dir+key+'_resnet.npy'),'rb')) for key in self.vids]
    # self.reg_fts = {}
    # reg_keys_ = glob(vid_dir+'Charades_region_feature/'+'*.npy')
    # reg_keys = [key.split('/')[-1][:-4] for key in reg_keys_]
    # for key in reg_keys:
    #   self.reg_fts[key] = pickle.load(open(os.path.join(vid_dir+'Charades_region_feature/',key+'.npy'),'rb'))

    # if 'VisualGenome' in name_file:
    #   global PIXEL_REDUCE
    #   PIXEL_REDUCE = 0

    # self.obj_ft_dir = obj_ft_dir
    # self.region_anno_dir = region_anno_dir

    # img_names = np.load(name_file)
    # self.img_id_to_ftidx_name = {x.split('.')[0]: (i, x) \
    #   for i, x in enumerate(img_names)}

    # self.mp_fts = np.load(mp_ft_file)
    # self.print_fn('mp_fts %s'%(str(self.mp_fts.shape)))

    # self.names = np.load(os.path.join(region_anno_dir, os.path.basename(name_file)))
    # self.num_data = len(self.names)
    self.num_data = len(self.data_ids)
    self.print_fn('num_data %d' % (self.num_data))

    # if pred_caption_file is None:
    #   self.pred_captions = None
    # else:
    #   self.pred_captions = json.load(open(pred_caption_file))

  def __getitem__(self, idx):
    image_id, region_id = self.names[idx]
    name = '%s_%s'%(image_id, region_id)

    anno = json.load(open(os.path.join(self.region_anno_dir, '%s.json'%image_id)))
    region_graph = anno[region_id]
    region_caption = anno[region_id]['phrase']

    with h5py.File(os.path.join(self.obj_ft_dir, '%s.jpg.hdf5'%image_id.replace('/', '_')), 'r') as f:
      key = '%s.jpg'%image_id.replace('/', '_')
      obj_fts = f[key][...]
      obj_bboxes = f[key].attrs['boxes']
      obj_box_to_ft = {tuple(box): ft for box, ft in zip(obj_bboxes, obj_fts)}

    attn_ft, node_types, attr_order_idxs = [], [], []
    obj_id_to_box = {}
    for x in region_graph['objects']:
      box = (x['x'], x['y'], x['x']+x['w']-PIXEL_REDUCE, x['y']+x['h']-PIXEL_REDUCE)
      obj_id_to_box[x['object_id']] = box
      attn_ft.append(obj_box_to_ft[box])
      attr_order_idxs.append(0)
      node_types.append(0)
      for ia, attr in enumerate(x['attributes']):
        attn_ft.append(obj_box_to_ft[box])
        attr_order_idxs.append(ia + 1)
        node_types.append(1)

    for x in region_graph['relationships']:
      obj_box = obj_id_to_box[x['object_id']]
      subj_box = obj_id_to_box[x['subject_id']]
      box = (min(obj_box[0], subj_box[0]), min(obj_box[1], subj_box[1]),
        max(obj_box[2], subj_box[2]), max(obj_box[3], subj_box[3]))
      attn_ft.append(obj_box_to_ft[box])
      node_types.append(2)
      attr_order_idxs.append(0)

    num_nodes = len(node_types)
    attn_ft, attn_mask = self.pad_or_trim_feature(
      np.array(attn_ft[:self.max_attn_len], np.float32),
      self.max_attn_len)
    node_types = node_types[:self.max_attn_len] + [0] * max(0, self.max_attn_len - num_nodes)
    node_types = np.array(node_types, np.int32)
    attr_order_idxs = attr_order_idxs[:self.max_attn_len] + [0] * max(0, self.max_attn_len - num_nodes)
    attr_order_idxs = np.array(attr_order_idxs, np.int32)

    out =  {
      'names': name,
      'mp_fts': self.mp_fts[self.img_id_to_ftidx_name[image_id][0]],
      'attn_fts': attn_ft,
      'attn_masks': attn_mask,
      'node_types': node_types,
      'attr_order_idxs': attr_order_idxs,
    } 
    if self.is_train or self.return_label:
      sent = region_caption
      caption_ids, caption_masks = self.pad_sents(self.sent2int(sent))
      out.update({
        'caption_ids': caption_ids,
        'caption_masks': caption_masks,
        'ref_sents': [sent],
        })
    # pdb.set_trace()
    return out

  def __len__(self):
    return self.num_data


def flat_collate_fn(data):
  outs = {}
  for key in ['names', 'mp_fts', 'attn_fts', 'attn_masks', 
              'caption_ids', 'caption_masks', 'node_types', 'attr_order_idxs']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]
      
  outs['mp_fts'] = np.array(outs['mp_fts'])
  max_attn_len = np.max(np.sum(outs['attn_masks'], 1))
  outs['attn_fts'] = np.array(outs['attn_fts'])[:, :max_attn_len]
  outs['attn_masks'] = np.array(outs['attn_masks'])[:, :max_attn_len]

  for key in ['node_types', 'attr_order_idxs']:
    if key in data[0]:
      outs[key] = np.array(outs[key])[:, :max_attn_len]

  if 'caption_ids' in data[0]:
    outs['caption_ids'] = np.array(outs['caption_ids'], np.int32)
    outs['caption_masks'] = np.array(outs['caption_masks'], np.bool)
    max_sent_len = np.max(np.sum(outs['caption_masks']))
    outs['caption_ids'] = outs['caption_ids'][:, :max_sent_len]
    outs['caption_masks'] = outs['caption_masks'][:, :max_sent_len]
    outs['ref_sents'] = {}
    for x in data:
      outs['ref_sents'][x['names']] = x['ref_sents']

  return outs

class ImageSceneGraphReader(ImageSceneGraphFlatReader):
  def add_obj_attr_edge(self, edges, obj_node_id, attr_node_id):
    edges.append([obj_node_id, attr_node_id, 0])
    edges.append([attr_node_id, obj_node_id, 1])

  def add_rel_subj_edge(self, edges, rel_node_id, subj_node_id):
    edges.append([subj_node_id, rel_node_id, 2])
    edges.append([rel_node_id, subj_node_id, 3])

  def add_rel_obj_edge(self, edges, rel_node_id, obj_node_i):
    edges.append([rel_node_id, obj_node_i, 4])
    edges.append([obj_node_i, rel_node_id, 5])

  def __getitem__(self, idx):
    image_id, region_id = self.names[idx]
    name = '%s_%s'%(image_id, region_id)
    anno = json.load(open(os.path.join(self.region_anno_dir, '%s.json'%image_id)))
    region_graph = anno[region_id]
    region_caption = anno[region_id]['phrase']

    with h5py.File(os.path.join(self.obj_ft_dir, '%s.jpg.hdf5'%image_id.replace('/', '_')), 'r') as f:
      key = '%s.jpg'%image_id.replace('/', '_')
      obj_fts = f[key][...]
      obj_bboxes = f[key].attrs['boxes']
      obj_box_to_ft = {tuple(box): ft for box, ft in zip(obj_bboxes, obj_fts)}

    attn_fts, node_types, attr_order_idxs, edges = [], [], [], []
    obj_id_to_box = {}
    obj_id_to_graph_id = {}
    n = 0
    for x in region_graph['objects']:
      box = (x['x'], x['y'], x['x']+x['w']-PIXEL_REDUCE, x['y']+x['h']-PIXEL_REDUCE)
      obj_id_to_box[x['object_id']] = box
      attn_fts.append(obj_box_to_ft[box])
      attr_order_idxs.append(0)
      node_types.append(0)
      obj_id_to_graph_id[x['object_id']] = n
      n += 1
      if n >= self.max_attn_len:
        break
      for ia, attr in enumerate(x['attributes']):
        attn_fts.append(obj_box_to_ft[box])
        attr_order_idxs.append(ia + 1)
        node_types.append(1)
        self.add_obj_attr_edge(edges, obj_id_to_graph_id[x['object_id']], n)
        n += 1
        if n >= self.max_attn_len:
          break
      if n >= self.max_attn_len:
        break

    if n < self.max_attn_len:
      for x in region_graph['relationships']:
        obj_box = obj_id_to_box[x['object_id']]
        subj_box = obj_id_to_box[x['subject_id']]
        box = (min(obj_box[0], subj_box[0]), min(obj_box[1], subj_box[1]),
          max(obj_box[2], subj_box[2]), max(obj_box[3], subj_box[3]))
        attn_fts.append(obj_box_to_ft[box])
        attr_order_idxs.append(0)
        node_types.append(2)
        self.add_rel_subj_edge(edges, n, obj_id_to_graph_id[x['subject_id']])
        self.add_rel_obj_edge(edges, n, obj_id_to_graph_id[x['object_id']])
        n += 1
        if n >= self.max_attn_len:
          break

    num_nodes = len(node_types)
    attn_fts = np.array(attn_fts, np.float32)
    attn_fts, attn_masks = self.pad_or_trim_feature(attn_fts, self.max_attn_len)
    node_types = node_types[:self.max_attn_len] + [0] * max(0, self.max_attn_len - num_nodes)
    node_types = np.array(node_types, np.int32)
    attr_order_idxs = attr_order_idxs[:self.max_attn_len] + [0] * max(0, self.max_attn_len - num_nodes)
    attr_order_idxs = np.array(attr_order_idxs, np.int32)

    if len(edges) > 0:
      src_nodes, tgt_nodes, edge_types = tuple(zip(*edges))
      src_nodes = np.array(src_nodes, np.int32)
      tgt_nodes = np.array(tgt_nodes, np.int32)
      edge_types = np.array(edge_types, np.int32)
      edge_counter = collections.Counter([(tgt_node, edge_type) for tgt_node, edge_type in zip(tgt_nodes, edge_types)])
      edge_norms = np.array(
        [1 / edge_counter[(tgt_node, edge_type)] for tgt_node, edge_type in zip(tgt_nodes, edge_types)],
          np.float32)
    else:
      tgt_nodes = src_nodes = edge_types = edge_norms = np.array([])

    edge_sparse_matrices = []
    for i in range(NUM_RELS):
      idxs = (edge_types == i)
      edge_sparse_matrices.append(
        sparse.coo_matrix((edge_norms[idxs], (tgt_nodes[idxs], src_nodes[idxs])), 
          shape=(self.max_attn_len, self.max_attn_len)))
    
    out =  {
      'names': name,
      'mp_fts': self.mp_fts[self.img_id_to_ftidx_name[image_id][0]],
      'attn_fts': attn_fts,
      'attn_masks': attn_masks,
      'node_types': node_types,
      'attr_order_idxs': attr_order_idxs,
      'edge_sparse_matrices': edge_sparse_matrices,
    } 
    if self.is_train or self.return_label:
      sent = region_caption
      caption_ids, caption_masks = self.pad_sents(self.sent2int(sent))
      out.update({
        'caption_ids': caption_ids,
        'caption_masks': caption_masks,
        'ref_sents': [sent],
        })
    return out


def sg_sparse_collate_fn(data):
  outs = {}
  for key in ['names', 'mp_fts', 'vid_len', 'attn_fts', 'vid_mask', 'attn_masks', 'node_types', 'attr_order_idxs', \
              'edge_sparse_matrices', 'caption_ids', 'caption_masks']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]
  
  outs['mp_fts'] = np.array(outs['mp_fts'])
  # outs['reg_fts'] = np.array(outs['reg_fts'])
  outs['vid_len'] = outs['vid_len']
  max_attn_len, dim_attn_ft = data[0]['attn_fts'].shape
  # (batch, max_attn_len, dim_attn_ft)
  outs['attn_fts'] = np.array(outs['attn_fts'], np.int32)
  outs['attn_masks'] = np.array(outs['attn_masks'])
  outs['vid_mask'] = np.array(outs['vid_mask'])

  if 'caption_ids' in data[0]:
    outs['caption_ids'] = np.array(outs['caption_ids'], np.int32)
    outs['caption_masks'] = np.array(outs['caption_masks'], np.bool)
    max_sent_len = np.max(np.sum(outs['caption_masks']))
    outs['caption_ids'] = outs['caption_ids'][:, :max_sent_len]
    outs['caption_masks'] = outs['caption_masks'][:, :max_sent_len]
    outs['ref_sents'] = {}
    for x in data:
      outs['ref_sents'][x['names']] = x['ref_sents']

  return outs


class ImageSceneGraphFlowReader(ImageSceneGraphReader):
  def __getitem__(self, idx):    
    data_id = self.data_ids[idx]
    # video_fts = self.video_fts[idx]
    vid = self.vids[idx]
    video_fts = np.load(os.path.join(self.vid_dir+vid+'_resnet.npy'))
    # video_fts = pickle.load(open(os.path.join(self.vid_dir+'Charades_feature/',vid+'.npy'),'rb'))

    region_graph = self.anno_data[data_id]
    region_caption = region_graph['phrase']

    node_types, attr_order_idxs = [], []
    attn_node_names = []
    # reg_fts = []
    edges, flow_edges = [], []
    obj_id_to_graph_id = {}
    n = 0
    for x in region_graph['objects']:
      attn_node_names.append(x['name'])
      # reg_fts.append(self.reg_fts[vid+'_'+x['name']])
      attr_order_idxs.append(0)
      node_types.append(0)
      obj_id_to_graph_id[x['object_id']] = n
      n += 1
      if n >= self.max_attn_len:
        break
      for ia, attr in enumerate(x['attributes']):
        attn_node_names.append(attr)
        # reg_fts.append(self.reg_fts[vid+'_'+x['name']])
        attr_order_idxs.append(ia + 1)
        node_types.append(1)
        self.add_obj_attr_edge(edges, obj_id_to_graph_id[x['object_id']], n)
        # bi-directional for obj-attr
        flow_edges.append((obj_id_to_graph_id[x['object_id']], n))
        flow_edges.append((n, obj_id_to_graph_id[x['object_id']]))
        n += 1
        if n >= self.max_attn_len:
          break
      if n >= self.max_attn_len:
        break

    # print(data_id)
    if n < self.max_attn_len:
      for x in region_graph['relationships']:
        attn_node_names.append(x['name'])
        attr_order_idxs.append(0)
        node_types.append(2)
        self.add_rel_subj_edge(edges, n, obj_id_to_graph_id[x['subject_id']])
        self.add_rel_obj_edge(edges, n, obj_id_to_graph_id[x['object_id']])
        flow_edges.append((obj_id_to_graph_id[x['subject_id']], n))
        flow_edges.append((n, obj_id_to_graph_id[x['object_id']]))
        # subject_feature = reg_fts[obj_id_to_graph_id[x['subject_id']]]
        # object_feature = reg_fts[obj_id_to_graph_id[x['object_id']]]
        # reg_fts.append(subject_feature + object_feature)
        n += 1
        if n >= self.max_attn_len:
          break
    video_fts, video_masks, vid_len = self.pad_or_trim_feature(video_fts, self.max_video_len)
    # reg_fts = np.array(reg_fts)
    # reg_fts, reg_masks, _ = self.pad_or_trim_feature(reg_fts, self.max_attn_len)

    num_nodes = len(node_types)
    node_types = node_types[:self.max_attn_len] + [0] * max(0, self.max_attn_len - num_nodes)
    node_types = np.array(node_types, np.int32)
    attr_order_idxs = attr_order_idxs[:self.max_attn_len] + [0] * max(0, self.max_attn_len - num_nodes)
    attr_order_idxs = np.array(attr_order_idxs, np.int32)

    if len(edges) > 0:
      src_nodes, tgt_nodes, edge_types = tuple(zip(*edges))
      src_nodes = np.array(src_nodes, np.int32)
      tgt_nodes = np.array(tgt_nodes, np.int32)
      edge_types = np.array(edge_types, np.int32)
      edge_counter = collections.Counter([(tgt_node, edge_type) for tgt_node, edge_type in zip(tgt_nodes, edge_types)])
      edge_norms = np.array(
        [1 / edge_counter[(tgt_node, edge_type)] for tgt_node, edge_type in zip(tgt_nodes, edge_types)],
          np.float32)
    else:
      tgt_nodes = src_nodes = edge_types = edge_norms = np.array([])

    # build python sparse matrix
    edge_sparse_matrices = []
    for i in range(NUM_RELS):
      idxs = (edge_types == i)
      edge_sparse_matrices.append(
        sparse.coo_matrix((edge_norms[idxs], (tgt_nodes[idxs], src_nodes[idxs])), 
                          shape=(self.max_attn_len, self.max_attn_len)))

    # add end flow loop
    # pdb.set_trace()
    flow_src_nodes = set([x[0] for x in flow_edges])
    for k in range(n):
      if k not in flow_src_nodes:
         flow_edges.append((k, k)) # end loop
    # flow order graph
    # if len(flow_edges) == 0:
    #   print(data_id)
    # pdb.set_trace()
    flow_src_nodes, flow_tgt_nodes = tuple(zip(*flow_edges))
    flow_src_nodes = np.array(flow_src_nodes, np.int32)
    flow_tgt_nodes = np.array(flow_tgt_nodes, np.int32)
    # normalize by src (collumn)
    flow_counter = collections.Counter(flow_src_nodes)
    flow_edge_norms = np.array(
      [1 / flow_counter[src_node] for src_node in flow_src_nodes])
    
    flow_sparse_matrix = sparse.coo_matrix((flow_edge_norms, (flow_tgt_nodes, flow_src_nodes)),
                                           shape=(self.max_attn_len, self.max_attn_len))

    node_name_ids = []

    for node in attn_node_names:
      node_name = self.sent2int(node)[:self.max_name_len]
      pad_num = self.max_name_len - len(node_name)
      node_name = node_name + [0]*pad_num
      node_name_ids.append(node_name)

    node_name_ids = np.array(node_name_ids)
    attn_fts, attn_mask , _= self.pad_or_trim_feature(node_name_ids, self.max_attn_len)


    # attn_mask = np.zeros((self.max_attn_len, ), np.bool)
    # seq_len = len(node_name_ids)
    # if seq_len < self.max_attn_len:
    #   attn_mask[:seq_len] = True
    # else:
    #   attn_mask[:] = True
    
    out =  {
      'names': data_id,
      'mp_fts': video_fts,
      # 'reg_fts': video_fts,
      # 'reg_fts': reg_fts,
      'vid_len': vid_len,
      'vid_mask': video_masks,
      'attn_fts': attn_fts,
      'attn_masks': attn_mask,
      'node_types': node_types,
      'attr_order_idxs': attr_order_idxs,
      'edge_sparse_matrices': edge_sparse_matrices,
      'flow_sparse_matrix': flow_sparse_matrix,
    } 

    # out =  {
    #   'names': name,
    #   'mp_fts': self.mp_fts[self.img_id_to_ftidx_name[image_id][0]],
    #   'attn_fts': attn_fts,
    #   'attn_masks': attn_masks,
    #   'node_types': node_types,
    #   'attr_order_idxs': attr_order_idxs,
    #   'edge_sparse_matrices': edge_sparse_matrices,
    #   'flow_sparse_matrix': flow_sparse_matrix,
    # } 
    if self.is_train or self.return_label:
      sent = region_caption
      caption_ids, caption_masks = self.pad_sents(self.sent2int(sent))
      out.update({
        'caption_ids': caption_ids,
        'caption_masks': caption_masks,
        'ref_sents': [sent],
        })
    
    
    return out


def sg_sparse_flow_collate_fn(data):
  outs = sg_sparse_collate_fn(data)
  outs['flow_sparse_matrix'] = [x['flow_sparse_matrix'] for x in data] 
  return outs
