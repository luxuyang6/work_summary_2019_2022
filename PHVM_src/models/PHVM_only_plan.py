import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import numpy as np
from onmt.decoders.transformer import TransformerDecoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.modules import Embeddings, VecEmbedding
import bottleneck as bn
import pdb
import random

np.set_printoptions(threshold=np.inf)  
torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None)


def rnn_mask(context_lens, max_step):
    """
    Creates a mask for variable length sequences
    """
    num_batches = len(context_lens)

    mask = torch.FloatTensor(num_batches, max_step).zero_()
    if torch.cuda.is_available():
        mask = mask.cuda()
    for b, batch_l in enumerate(context_lens):
        mask[b, :batch_l] = 1.0
    mask = Variable(mask)
    return mask

def sample_gaussian(shape, mu, logvar):
    x = Variable(torch.randn(shape)).cuda()
    return mu + torch.exp(logvar/2) * x


def KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar, loss_mask=None):
    divergence = 0.5 * torch.sum(torch.exp(post_logvar - prior_logvar)
                                        + torch.pow(post_mu - prior_mu, 2) / torch.exp(prior_logvar)
                                        - 1 - (post_logvar - prior_logvar), dim=1)
    if loss_mask is not None:
        return torch.sum(loss_mask.cuda() * divergence)
    else:
        return torch.sum(divergence)

def top_n_indexes(arr, n):
        idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
        width = arr.shape[1]
        return [divmod(i, width) for i in idx]



class PHVM_GRU(nn.Module):
    def __init__(self, args):
        super(PHVM_GRU, self).__init__()
        self.args = args

        self.FramesEncoder = FramesEncoder(self.args)
        self.KeywordsEncoder = WordsEncoder(self.args, self.args.enc_dim, self.args.enc_layer)
        self.TextsEncoder = WordsEncoder(self.args, self.args.text_encoder_dim, self.args.text_encoder_layer)
        self.GroupDecoder = GroupDecoder(self.args)
        #self.Decoder = Decoder(self.args)

        self.prior_linear1 = nn.Linear(self.args.vid_hid*2+self.args.enc_dim*2, self.args.plan_latent_dim*2)
        self.prior_linear2 = nn.Linear(self.args.plan_latent_dim*2, self.args.plan_latent_dim*2)
        self.post_linear1 = nn.Linear(self.args.vid_hid*2+self.args.enc_dim*2+self.args.text_encoder_dim*2, self.args.plan_latent_dim*2)
        self.post_linear2 = nn.Linear(self.args.plan_latent_dim*2, self.args.plan_latent_dim*2)


    def forward(self, frames, frame_lens, keywords, keyword_lens, texts, texts_lens, segments, group_count):
        batch_size = frames.shape[0]
        #print(keywords.shape,texts.shape)
        video_features = self.FramesEncoder(frames, frame_lens)
        video_features_t = Variable(torch.zeros(video_features.shape[0],video_features.shape[2])).cuda()
        for i in range(batch_size):
            video_features_t[i] = video_features[i,frame_lens[i]-1,:]  # get last state
        
        keywords_encode = self.KeywordsEncoder(keywords,keyword_lens)
        keywords_encode_t = Variable(torch.zeros(keywords_encode.shape[0],keywords_encode.shape[2])).cuda()
        for i in range(batch_size):
            keywords_encode_t[i] = keywords_encode[i,keyword_lens[i]-1,:]  # get last state

        text_encode = self.TextsEncoder(texts,texts_lens)
        text_encode_t = Variable(torch.zeros(text_encode.shape[0],text_encode.shape[2])).cuda()
        for i in range(batch_size):
            text_encode_t[i] = text_encode[i,texts_lens[i]-1,:]  # get last state

        post_mu , post_logvar = self.get_post_latent(keywords_encode_t, text_encode_t, video_features_t)
        prior_mu , prior_logvar = self.get_prior_latent(keywords_encode_t, video_features_t)
        global_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
        
        # 串联global_z和keywords_encode_t，线性层，作为group_init_state
        group_init_state = torch.cat((keywords_encode_t, video_features_t, global_z), 1)
        stop_loss, group_loss = self.GroupDecoder(keywords_encode, segments, group_count, group_init_state)

        #bow_loss, local_KL, word_loss = self.Decoder(keywords_encode, video_features, frame_lens, segments, group_count, captions, caption_lens, group_init_state)


        global_KL = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)
        #global_KL = KL_divergence(post_mu, post_logvar, prior_mu, prior_logvar)

        #loss = stop_loss + word_loss + group_loss + global_KL + local_KL + bow_loss


        return stop_loss , group_loss , global_KL


    def get_prior_latent(self, keywords_encode_t, video_features_t):

        prior_inp = torch.cat((keywords_encode_t, video_features_t), 1)
        prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.args.plan_latent_dim,1)
        return prior_mu , prior_logvar

    def get_post_latent(self, keywords_encode_t, text_encode_t, video_features_t):

        post_inp = torch.cat((keywords_encode_t, video_features_t, text_encode_t), 1)
        post_mu , post_logvar = torch.split(self.post_linear2(F.tanh(self.post_linear1(post_inp))),self.args.plan_latent_dim,1)
        return post_mu , post_logvar

    def shuffle_plan(self, segments_gt,group_count_gt):

        segments = torch.zeros(segments_gt.shape) 
        batch_size,step,_ = segments_gt.shape
        for i in range(batch_size):
            segment = segments_gt[i]
            index = [*range(group_count_gt[i])]
            random.shuffle(index)
            index = index+[*range(group_count_gt[i],step)]
            segment = segment.index_select(0, torch.tensor(index).cuda())
            segments[i] = segment

        return segments.cuda()

    def _agg_group_plan(self, group_count, segments,keywords):

        plans = []
        for gcnt,segs,keys in zip(group_count,segments, keywords):
            segs = segs[:gcnt]
            plan = []
            for segIG, seg in enumerate(segs):
                for i in range(len(seg)):
                    if seg[i] == 0:
                        continue
                    else:
                        plan.append(keys[i].cpu().item())
                plan.append(2)
            plans.append(plan)

        return plans

    def full_plan(self, segments_gt,group_count_gt):
        #pdb.set_trace()
        segments = torch.zeros(segments_gt.shape) 
        batch_size,_,_ = segments_gt.shape
        for i in range(batch_size):
            seg = torch.LongTensor(np.zeros(segments_gt.shape[-1])).cuda() 
            for j in range(group_count_gt[i]):
                seg = seg | segments_gt[i,j]
            for j in range(group_count_gt[i]):
                segments[i,j] = seg
        print('segments_gt',segments_gt)
        print('segments',segments)

        return segments.cuda()

    def random_plan(self, keyword_lens):
        
        batch_size = keyword_lens.shape[0]
        step_size = self.args.random_max_sent_len
        max_enc_len = torch.max(keyword_lens)
        segments = Variable(torch.LongTensor(np.zeros([batch_size,step_size,max_enc_len]))).cuda()
        group_count = torch.LongTensor(np.zeros(batch_size)).cuda()
        for i in range(batch_size):
            stop_len = random.randint(self.args.random_min_sent_len,self.args.random_max_sent_len)
            keyword_list = [*range(keyword_lens[i])]
            group_count[i] = stop_len
            for j in range(stop_len):
                random.shuffle(keyword_list)
                word_count = random.randint(self.args.random_min_word_len,self.args.random_max_word_len)
                choice = keyword_list[:word_count]
                for k in choice:
                    segments[i,j,k] = 1

        return segments, group_count 







    def sample(self, frames, flengths):
        video_features = self.encoder.forward(frames, flengths)
        predicted_target = self.decoder.sample(video_features, flengths)
        return predicted_target

    def sample_rl(self, frames, flengths, sampling='multinomial'):
        video_features = self.encoder.forward(frames, flengths)
        predicted_target, outputs = self.decoder.rl_sample(video_features, flengths, sampling=sampling)
        return predicted_target, outputs

    def beam_search(self, frames, frame_lens, keywords, keyword_lens, segments_gt, group_count_gt):
        batch_size = frames.shape[0]
        #print(keywords.shape,texts.shape)
        video_features = self.FramesEncoder(frames, frame_lens)
        video_features_t = Variable(torch.zeros(video_features.shape[0],video_features.shape[2])).cuda()
        for i in range(batch_size):
            video_features_t[i] = video_features[i,frame_lens[i]-1,:]  # get last state
        
        keywords_encode = self.KeywordsEncoder(keywords,keyword_lens)
        keywords_encode_t = Variable(torch.zeros(keywords_encode.shape[0],keywords_encode.shape[2])).cuda()
        for i in range(batch_size):
            keywords_encode_t[i] = keywords_encode[i,keyword_lens[i]-1,:]  # get last state

        prior_mu , prior_logvar = self.get_prior_latent(keywords_encode_t, video_features_t)
        global_z = sample_gaussian(prior_mu.shape, prior_mu , prior_logvar)
        
        # 串联global_z和keywords_encode_t，线性层，作为group_init_state
        group_init_state = torch.cat((keywords_encode_t, video_features_t, global_z), 1)
        if self.args.use_gt_plan:
            segments_p = segments_gt
            group_count_p =  group_count_gt
            #print('Use GT plan')
        elif self.args.random_plan:
            segments_p, group_count_p = self.random_plan(keyword_lens)
            #print('Use random plan')
        elif self.args.shuffle_plan:
            segments_p = self.shuffle_plan(segments_gt,group_count_gt)
            group_count_p =  group_count_gt
            #print('Use shuffle plan')
        elif self.args.full_plan:
            segments_p = self.full_plan(segments_gt,group_count_gt)
            group_count_p =  group_count_gt
            #print('Use full plan')
        else:
            segments_p, group_count_p = self.GroupDecoder.sample(batch_size, keywords_encode, group_init_state)
            #print('Use Generated plan')
        #pdb.set_trace()
        used_plans = self._agg_group_plan(group_count_p,segments_p, keywords)
        gt_plans = self._agg_group_plan(group_count_gt,segments_gt, keywords)
        #plans, output_ids = self.Decoder.beam_search(keywords, keywords_encode, video_features, frame_lens, segments, group_count,  group_init_state, self.args.beam_size)
        #used_plans, gt_plans, output_ids = self.Decoder.beam_search(keywords, keywords_encode, video_features, \
        #frame_lens, segments_p, group_count_p, segments_gt, group_count_gt,  group_init_state, self.args.beam_size)
        return used_plans, gt_plans


# Based on tutorials/08 - Language Model
# RNN Based Language Model
class FramesEncoder(nn.Module):
    def __init__(self, args):
        super(FramesEncoder, self).__init__()

        self.args = args
        self.embed_size = self.args.embed_size
        self.vid_dim = self.args.vid_dim
        self.hidden_dim = self.args.vid_hid
        self.enable_cuda = self.args.cuda
        self.layers = self.args.vid_layer
        self.dropout_rate = self.args.dropout
        self.birnn = self.args.birnn

        self.linear = nn.Linear(self.vid_dim, self.embed_size, bias=False)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_dim, self.layers, batch_first=True, bidirectional=self.birnn, dropout=self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        

    def init_hidden(self, batch_size):
        if self.birnn:
            return (Variable(torch.zeros(2*self.layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2*self.layers, batch_size, self.hidden_dim)))



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
            frames = frames.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
        else:
            frames = frames.index_select(0, Variable(torch.LongTensor(idx_sort)))



        frames = self.linear(frames)
        #frame_packed = nn.utils.rnn.pack_padded_sequence(frames, flengths, batch_first=True)
        frame_packed = nn.utils.rnn.pack_padded_sequence(frames, flengths.copy(), batch_first=True)
        outputs, (ht, ct) = self.rnn(frame_packed, self.init_rnn)
        outputs,_ = pad_packed_sequence(outputs,batch_first=True)

        idx_unsort = np.argsort(idx_sort)
        if self.enable_cuda:
            outputs = outputs.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)))
        else:
            outputs = outputs.index_select(0, Variable(torch.LongTensor(idx_unsort)))

        # print 'Encoder Outputs:',outputs.size()

        return outputs



# Based on tutorials/08 - Language Model
# RNN Based Language Model
class WordsEncoder(nn.Module):
    def __init__(self, args, dim, layers, birnn=True):
        super(WordsEncoder, self).__init__()

        # self.use_abs = use_abs
        self.args = args
        self.embed_size = self.args.embed_size
        self.vocab_size = self.args.vocab_size
        self.hidden_dim = dim
        self.enable_cuda = self.args.cuda
        self.layers = layers
        self.dropout_rate = self.args.dropout
        self.birnn = birnn
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        self.rnn = nn.GRU(self.embed_size, self.hidden_dim, self.layers, batch_first=True, bidirectional=self.birnn, dropout=self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)

    def init_hidden(self, batch_size):
        if self.birnn:
            return (Variable(torch.zeros(2*self.layers, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2*self.layers, batch_size, self.hidden_dim)))



    def forward(self, inputs, flengths):
        """Handles variable size inputs
           frame_embed: video features
           flengths: frame lengths
        """
        batch_size = inputs.shape[0]
        #inputs = self.linear(inputs)
        #inputs = self.dropout(inputs) # adding dropout layer
        self.init_rnn = self.init_hidden(batch_size)[0]
        if self.enable_cuda:
            self.init_rnn = self.init_rnn.cuda()

        flengths, idx_sort = np.sort(flengths.cpu().numpy())[::-1], np.argsort(-flengths.cpu().numpy())
        if self.enable_cuda:
            inputs = inputs.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
        else:
            inputs = inputs.index_select(0, Variable(torch.LongTensor(idx_sort)))



        embeddings = self.embed(inputs)
        #frame_packed = nn.utils.rnn.pack_padded_sequence(inputs, flengths, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, flengths.copy(), batch_first=True)
        outputs, ht = self.rnn(packed, self.init_rnn)
        outputs,_ = pad_packed_sequence(outputs,batch_first=True)

        idx_unsort = np.argsort(idx_sort)
        if self.enable_cuda:
            outputs = outputs.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)))
        else:
            outputs = outputs.index_select(0, Variable(torch.LongTensor(idx_unsort)))

        # print 'Encoder Outputs:',outputs.size()

        return outputs

class GroupEncoder(nn.Module):
    def __init__(self, args):
        super(GroupEncoder, self).__init__()

        # self.use_abs = use_abs
        self.args = args

        self.enable_cuda = self.args.cuda
        self.layers = self.args.group_encoder_layer
        self.dropout_rate = self.args.dropout
        self.enc_dim = self.args.group_encoder_dim

        self.rnn = nn.GRU(self.args.enc_dim*2, self.args.group_encoder_dim, self.layers, batch_first=True, dropout=self.dropout_rate)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.birnn*self.layers, batch_size, self.enc_dim)),
                Variable(torch.zeros(self.birnn*self.layers, batch_size, self.enc_dim)))



    def forward(self, enc_states, segments, group_count):
        """Handles variable size inputs
           frame_embed: video features
           flengths: frame lengths
        """
        batch_size,step_size,_ = segments.shape

        group_count, idx_sort = np.sort(group_count.cpu().numpy())[::-1], np.asarray(np.argsort(-group_count.cpu().numpy()))
        if self.enable_cuda:
            
            #pdb.set_trace()
            segments = segments.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
        else:
            segments = segments.index_select(0, Variable(torch.LongTensor(idx_sort)))

        inps = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,self.enc_dim*2]))).cuda()
        for i in range(step_size):
            segment = segments[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inps[:,i,:] = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)


        state = None

        inps_packed = nn.utils.rnn.pack_padded_sequence(inps, group_count.copy(), batch_first=True)
        outputs, ht = self.rnn(inps_packed, state)

        outputs,_ = pad_packed_sequence(outputs,batch_first=True)

        idx_unsort = np.argsort(idx_sort)
        if self.enable_cuda:
            outputs = outputs.index_select(0, Variable(torch.cuda.LongTensor(idx_unsort)))
        else:
            outputs = outputs.index_select(0, Variable(torch.LongTensor(idx_unsort)))

        return outputs


class GroupDecoder(nn.Module):
    def __init__(self, args):
        super(GroupDecoder, self).__init__()

        # self.use_abs = use_abs
        self.args = args

        self.hidden_dim = self.args.group_decoder_dim
        self.enc_dim = self.args.enc_dim
        self.vid_hid = self.args.vid_hid
        self.max_group_count = self.args.max_group_count
        self.plan_latent_dim = self.args.plan_latent_dim
        self.dropout_rate = self.args.dropout

        self.enable_cuda = self.args.cuda
        
        self.rnn = nn.GRU(self.enc_dim*2, self.hidden_dim, batch_first=True, dropout=self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.initLinear = nn.Linear(self.enc_dim*2+self.vid_hid*2+self.plan_latent_dim, self.hidden_dim)

        self.PlanLinear1 = nn.Linear(self.hidden_dim+self.enc_dim*2, self.enc_dim)
        self.PlanLinear2 = nn.Linear(self.enc_dim, 1)
        self.StopLinear = nn.Linear(self.hidden_dim, 1)
        
        

    def forward(self, enc_states, segments, group_count, group_init_state):
        """Decode image feature vectors and generates captions."""
        """
        :param video_features:
            video encoder output hidden states of size batch_size x max_enc_steps x hidden_dim
        :param flengths:
            video frames length of size batch_size
        :param captions:
            input target captions of size batch_size x max_dec_steps
        :param lengths:
            input captions lengths of size batch_size

        """
        # print features.size(), captions.size(), self.embed_size
        # print 'Input features, captions, lengths', features.size(), captions.size(), lengths, np.sum(lengths)
        # appending <start> token to the input captions
        batch_size,step_size,_ = segments.shape
        self.max_enc_len = enc_states.shape[1]
        max_enc_len = self.max_enc_len
        group_count, idx_sort = np.sort(group_count.cpu().numpy())[::-1], np.argsort(-group_count.cpu().numpy())
        if self.enable_cuda:
            segments = segments.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
        else:
            segments = segments.index_select(0, Variable(torch.LongTensor(idx_sort)))

        segments_input = torch.cat((Variable(torch.LongTensor(np.zeros([batch_size,1,max_enc_len]))).cuda(),segments), 1) #key start group
        inps = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,self.enc_dim*2]))).cuda()
        for i in range(step_size):
            segment = segments_input[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inps[:,i,:] = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)


        state = self.initLinear(group_init_state).unsqueeze(0)

        inps_packed = nn.utils.rnn.pack_padded_sequence(inps, group_count.copy(), batch_first=True)
        outputs, ht = self.rnn(inps_packed, state)
        outputs = outputs[0] #取pack的数据
        #outputs,_ = pad_packed_sequence(outputs,batch_first=True)
        group_targets_packed = nn.utils.rnn.pack_padded_sequence(segments, group_count.copy(), batch_first=True)[0]
    
         #key data_len * max_enc_len
        enc_state_2 = nn.utils.rnn.pack_padded_sequence(enc_states.unsqueeze(1).expand(-1,step_size,-1,-1), group_count.copy(), batch_first=True)[0] #key packed_data_len * max_enc_len * enc_dim
        enc_state_2 = torch.cat((enc_state_2, Variable(torch.FloatTensor(np.zeros([enc_state_2.shape[0],max_enc_len-enc_state_2.shape[1],self.enc_dim*2]))).cuda()),1)
        outputs_2 = outputs.unsqueeze(1).expand(-1,max_enc_len,-1) #key packed_data_len * max_enc_len * hidden_dim
        outputs_2 = torch.cat((outputs_2, enc_state_2),2)

        stop_label = Variable(torch.zeros(batch_size, step_size)).cuda()
        for i in range(batch_size):
            stop_label[i,group_count[i] -1] = 1

        stop_packed = nn.utils.rnn.pack_padded_sequence(stop_label, group_count.copy(), batch_first=True)[0]

        group_loss = F.binary_cross_entropy_with_logits(self.PlanLinear2(F.tanh(self.PlanLinear1(outputs_2))).squeeze(2), group_targets_packed.float())
        stop_loss = F.binary_cross_entropy_with_logits(self.StopLinear(outputs).squeeze(1), stop_packed.float()) 

        return stop_loss, group_loss

    def sample(self, batch_size, enc_states, group_init_state):

        batch_size,max_enc_len,_ = enc_states.shape
        self.max_enc_len = enc_states.shape[1]
        step_size = self.args.max_group_count

        inp = Variable(torch.FloatTensor(np.zeros([batch_size,1,self.enc_dim*2]))).cuda()
        state = self.initLinear(group_init_state).unsqueeze(0)
        stop_pred = Variable(torch.FloatTensor(np.zeros([batch_size,step_size]))).cuda()
        segments = Variable(torch.LongTensor(np.zeros([batch_size,step_size,max_enc_len]))).cuda()
        group_count = torch.LongTensor(np.zeros(batch_size))
        for i in range(step_size):

            outputs, state = self.rnn(inp, state)
            enc_state_2 = torch.cat((enc_states, Variable(torch.FloatTensor(np.zeros([enc_states.shape[0],max_enc_len-enc_states.shape[1],self.enc_dim*2]))).cuda()),1)
            outputs_2 = outputs.expand(-1,max_enc_len,-1) #key batch_size * max_enc_len * hidden_dim
            outputs_2 = torch.cat((outputs_2, enc_state_2),2)

            plan_pred = F.sigmoid(self.PlanLinear2(F.tanh(self.PlanLinear1(outputs_2))).squeeze(2))  # batch_size*max_enc_len
            stop_pred[:,i] = F.sigmoid(self.StopLinear(outputs.squeeze(1)).squeeze(1)) # batch*step
            
            # pdb.set_trace()
            segment = Variable((plan_pred > self.args.group_selection_threshold).long()).cuda()
            for j in range(batch_size):
                if torch.sum(segment[j]) == 0:
                    segment[j][torch.argmax(plan_pred[j])] = 1
            
            segments[:,i,:] = segment
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inp = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states) # (batch_size, 1, enc_dim)
        
        for j in range(batch_size):
            if len((stop_pred[j]>self.args.stop_threshold).nonzero()) != 0:
                group_count[j] = (stop_pred[j]>self.args.stop_threshold).nonzero()[0] + 1 #第一个大于0.5的值的索引 + 1
            else:
                group_count[j] = torch.argmax(stop_pred[j]) + 1
        #print('group_count:',group_count)

        return segments, group_count  # batch_size*max_step*max_keyword_len, array (batch_size*1)
