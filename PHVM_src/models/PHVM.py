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
        self.Decoder = Decoder(self.args)

        self.prior_linear1 = nn.Linear(self.args.vid_hid*2+self.args.enc_dim*2, self.args.plan_latent_dim*2)
        self.prior_linear2 = nn.Linear(self.args.plan_latent_dim*2, self.args.plan_latent_dim*2)
        self.post_linear1 = nn.Linear(self.args.vid_hid*2+self.args.enc_dim*2+self.args.text_encoder_dim*2, self.args.plan_latent_dim*2)
        self.post_linear2 = nn.Linear(self.args.plan_latent_dim*2, self.args.plan_latent_dim*2)


    def forward(self, frames, frame_lens, keywords, keyword_lens, texts, texts_lens, segments, group_count,segments_gt, group_count_gt, captions, caption_lens, anneal):
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

        # stop_loss, group_loss = self.GroupDecoder(keywords_encode, segments_gt, group_count_gt, group_init_state)

        # segments, segments_soft, group_count = self.GroupDecoder.gumbel_sample(batch_size, keywords_encode, group_init_state)
        bow_loss, local_KL, word_loss = self.Decoder(keywords_encode, video_features, frame_lens, segments, group_count, \
                                                    segments_gt, group_count_gt, captions, caption_lens, group_init_state, anneal)


        global_KL = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)

        #loss = stop_loss + word_loss + group_loss + global_KL + local_KL + bow_loss


        # return stop_loss, group_loss, word_loss , global_KL , local_KL , bow_loss
        return word_loss , global_KL , local_KL , bow_loss


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

    def beam_search(self, frames, frame_lens, keywords, keyword_lens, segments, group_count, segments_gt, group_count_gt):
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
            segments = segments_gt
            group_count =  group_count_gt
        #     #print('Use GT plan')
        # elif self.args.random_plan:
        #     segments_p, group_count_p = self.random_plan(keyword_lens)
        #     #print('Use random plan')
        # elif self.args.shuffle_plan:
        #     segments_p = self.shuffle_plan(segments_gt,group_count_gt)
        #     group_count_p =  group_count_gt
        #     #print('Use shuffle plan')
        # elif self.args.full_plan:
        #     segments_p = self.full_plan(segments_gt,group_count_gt)
        #     group_count_p =  group_count_gt
        #     #print('Use full plan')
        else:
            segments, group_count = self.GroupDecoder.sample(batch_size, keywords_encode, group_init_state)
        #     # print('Use Generated plan')
        #plans, output_ids = self.Decoder.beam_search(keywords, keywords_encode, video_features, frame_lens, segments, group_count,  group_init_state, self.args.beam_size)
        used_plans, gt_plans, output_ids = self.Decoder.beam_search(keywords, keywords_encode, video_features, \
        frame_lens, segments, group_count, segments_gt, group_count_gt,  group_init_state, self.args.beam_size)
        return used_plans, gt_plans, output_ids


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



    def forward(self, enc_states, segments, group_count, segments_gt, group_count_gt, anneal):
        """Handles variable size inputs
           frame_embed: video features
           flengths: frame lengths
        """
        batch_size,step_size,keywords_len = segments.shape

        group_count, idx_sort = np.sort(group_count.cpu().numpy())[::-1], np.asarray(np.argsort(-group_count.cpu().numpy()))
        if self.enable_cuda:
            
            #pdb.set_trace()
            segments = segments.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            segments_gt = segments_gt.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
        else:
            segments = segments.index_select(0, Variable(torch.LongTensor(idx_sort)))
            segments_gt = segments_gt.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))

        inps = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,self.enc_dim*2]))).cuda()
        for i in range(step_size):
            j = segments_gt.shape[1]
            segment = segments[:,i,:]
            # pdb.set_trace()
            if i < j: 
                segment_gt = segments_gt[:,i,:]
            else:
                segment_gt = Variable(torch.LongTensor(np.zeros([batch_size, keywords_len]))).cuda()
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            alpha_gt = torch.div(segment_gt.float(), torch.add(segment_gt.sum(1).unsqueeze(1).expand_as(segment_gt),1).float()) #key 归一化
            gbow_ = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
            gbow_gt = torch.bmm(alpha_gt[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
            #inps[:,i,:] = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
            inps[:,i,:] = anneal * gbow_ + (1.0 - anneal) * gbow_gt


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
        self.eps = 1e-20
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
            
            #pdb.set_trace()
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

    def gumbel_sample(self, batch_size, enc_states, group_init_state):

        batch_size,max_enc_len,_ = enc_states.shape
        self.max_enc_len = enc_states.shape[1]
        step_size = self.args.max_group_count

        inp = Variable(torch.FloatTensor(np.zeros([batch_size,1,self.enc_dim*2]))).cuda()
        state = self.initLinear(group_init_state).unsqueeze(0)
        stop_pred = Variable(torch.FloatTensor(np.zeros([batch_size,step_size]))).cuda()
        segments = Variable(torch.LongTensor(np.zeros([batch_size,step_size,max_enc_len]))).cuda()
        segments_soft = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,max_enc_len]))).cuda()
        group_count = torch.LongTensor(np.zeros(batch_size))
        for i in range(step_size):

            outputs, state = self.rnn(inp, state)
            enc_state_2 = torch.cat((enc_states, Variable(torch.FloatTensor(np.zeros([enc_states.shape[0],max_enc_len-enc_states.shape[1],self.enc_dim*2]))).cuda()),1)
            outputs_2 = outputs.expand(-1,max_enc_len,-1) #key batch_size * max_enc_len * hidden_dim
            outputs_2 = torch.cat((outputs_2, enc_state_2),2)
            plan_logits = self.PlanLinear2(F.tanh(self.PlanLinear1(outputs_2))).squeeze(2)
            plan_pred = F.sigmoid(plan_logits)  # batch_size*max_enc_len
            stop_pred[:,i] = F.sigmoid(self.StopLinear(outputs.squeeze(1)).squeeze(1)) # batch*step
            uniform1 = torch.rand(plan_logits.shape).cuda()
            uniform2 = torch.rand(plan_logits.shape).cuda()
            gumbel_noise = -torch.log(torch.log(uniform2 + self.eps)/torch.log(uniform1 + self.eps) +self.eps)
            segment_soft = F.sigmoid((plan_logits + gumbel_noise)/self.args.gumbel_temperature) 
            #pdb.set_trace()
            segment = Variable((plan_pred > self.args.group_selection_threshold).long()).cuda()
            for j in range(batch_size):
                if torch.sum(segment[j]) == 0:
                    segment[j][torch.argmax(plan_pred[j])] = 1
            
            segments[:,i,:] = segment
            segments_soft[:,i,:] = segment_soft
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inp = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states) # (batch_size, 1, enc_dim)
            # 这里算alpha和inp是不是应该放在stop之后，得把stop之后的segments变成pad 
        for j in range(batch_size):
            if len((stop_pred[j]>self.args.stop_threshold).nonzero()) != 0:
                group_count[j] = (stop_pred[j]>self.args.stop_threshold).nonzero()[0] + 1 #第一个大于0.5的值的索引 + 1
            else:
                group_count[j] = torch.argmax(stop_pred[j]) + 1
        #print('group_count:',group_count)

        return segments, segments_soft, group_count  # batch_size*max_step*max_keyword_len, array (batch_size*1)



class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        # self.use_abs = use_abs
        self.args = args
        self.group_encoder_dim = self.args.group_encoder_dim
        self.vid_hid = self.args.vid_hid

        self.sent_latent_dim = self.args.sent_latent_dim

        self.enc_dim = self.args.enc_dim
        self.decoder_dim = self.args.decoder_dim
        self.decoder_layer = self.args.decoder_layer
        self.bow_hid = self.args.bow_hid
        self.plan_latent_dim = self.args.plan_latent_dim

        self.sent_decoder_dim = self.args.sent_decoder_dim 

        self.embed_size = self.args.embed_size
        self.vocab_size = self.args.vocab_size

        self.enable_cuda = self.args.cuda
        self.dropout_rate = self.args.dropout
        
        self.use_gt_plan = self.args.use_gt_plan
        self.SentEncoder = WordsEncoder(self.args, self.args.text_encoder_dim, self.args.text_encoder_layer)
        self.GroupEncoder = GroupEncoder(self.args)

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        if self.args.plan_dim_decay!= self.enc_dim:
            self.decayLinear =  nn.Linear(self.enc_dim*2, self.args.plan_decay_dim*2)
            self.bowLinear1 = nn.Linear(self.args.sent_latent_dim+self.sent_decoder_dim+self.args.plan_decay_dim*2, self.bow_hid)
            self.initLinear = nn.Linear(self.args.plan_decay_dim*2+self.vid_hid*2+self.plan_latent_dim+self.group_encoder_dim, self.sent_decoder_dim)
            self.initLinear2 = nn.Linear(self.args.sent_latent_dim+self.sent_decoder_dim+self.args.plan_decay_dim*2, self.decoder_dim)
            self.prior_linear1 = nn.Linear(self.args.sent_decoder_dim+self.args.plan_decay_dim*2, self.args.sent_latent_dim*2)
            self.post_linear1 = nn.Linear(self.args.sent_decoder_dim+self.args.plan_decay_dim*2+self.args.text_encoder_dim*2, self.args.sent_latent_dim*2)

        else:
            self.bowLinear1 = nn.Linear(self.args.sent_latent_dim+self.sent_decoder_dim+self.enc_dim*2, self.bow_hid)
            self.initLinear = nn.Linear(self.enc_dim*2+self.vid_hid*2+self.plan_latent_dim+self.group_encoder_dim, self.sent_decoder_dim)
            self.initLinear2 = nn.Linear(self.args.sent_latent_dim+self.sent_decoder_dim+self.enc_dim*2, self.decoder_dim)
            self.prior_linear1 = nn.Linear(self.args.sent_decoder_dim+self.args.enc_dim*2, self.args.sent_latent_dim*2)
            self.post_linear1 = nn.Linear(self.args.sent_decoder_dim+self.args.enc_dim*2+self.args.text_encoder_dim*2, self.args.sent_latent_dim*2)
        
        self.prior_linear2 = nn.Linear(self.args.sent_latent_dim*2, self.args.sent_latent_dim*2)
        self.post_linear2 = nn.Linear(self.args.sent_latent_dim*2, self.args.sent_latent_dim*2)
        self.bowLinear2 = nn.Linear(self.bow_hid, self.vocab_size)
        self.wordLinear = nn.Linear(self.decoder_dim, self.vocab_size)

        self.wordRnn = nn.GRU(self.embed_size+self.vid_hid*2, self.decoder_dim, self.decoder_layer, batch_first=True, dropout=self.dropout_rate)
        self.sentRnn = nn.GRU(self.decoder_dim+self.sent_latent_dim, self.sent_decoder_dim, batch_first=True, dropout=self.dropout_rate)
        self.atten = Attention(self.args, self.vid_hid*2, self.decoder_dim)
        
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, enc_states, video_features, frame_lens, segments, group_count, segments_gt, group_count_gt, captions, caption_lens, group_init_state, anneal):

        batch_size,step_size,max_dec_len = captions.shape

        group_encode = self.GroupEncoder(enc_states, segments, group_count, segments_gt, group_count_gt, anneal)

        group_encode_t = Variable(torch.zeros(group_encode.shape[0],group_encode.shape[2])).cuda()
        for i in range(batch_size):
            group_encode_t[i] = group_encode[i, group_count[i]-1,:]  # get last state
        
        state = self.initLinear(torch.cat((group_init_state, group_encode_t), 1)).unsqueeze(0)
        inp = Variable(torch.zeros(batch_size,self.decoder_dim+self.sent_latent_dim)).cuda().unsqueeze(1)
        _captions = torch.cat((Variable(torch.LongTensor(np.ones([batch_size,step_size,1]))).cuda(),captions), 2)
        embeddings = self.embed(_captions)
        context_mask = rnn_mask(frame_lens, video_features.shape[1])

        for i in range(step_size):

            sent_hidden,state = self.sentRnn(inp,state)
            #print('sent_hidden:',sent_hidden, 'state:',state)
            hidden_output = sent_hidden.squeeze(1)
            if i < segments.shape[1]: 
                segment = segments[:,i,:]
            else:
                segment = Variable(torch.LongTensor(np.zeros([batch_size, segments.shape[2]]))).cuda()
            segment_gt = segments_gt[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            alpha_gt = torch.div(segment_gt.float(), torch.add(segment_gt.sum(1).unsqueeze(1).expand_as(segment_gt),1).float()) #key 归一化
            gbow_ = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
            gbow_gt = torch.bmm(alpha_gt[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
            gbow = anneal * gbow_ + (1.0 - anneal) * gbow_gt
            # if self.args.plan_dim_decay!= self.enc_dim:
            #    gbow = self.decayLinear(gbow) 
            

            sent_encode = self.SentEncoder(captions[:,i,:],caption_lens[:,i])
            sent_encode_t = Variable(torch.zeros(sent_encode.shape[0],sent_encode.shape[2])).cuda()
            for j in range(batch_size):
                sent_encode_t[j] = sent_encode[j, caption_lens[j,i]-1,:]  # get the last state
            post_mu , post_logvar = self.get_post_latent(hidden_output, sent_encode_t, gbow)
            prior_mu , prior_logvar = self.get_prior_latent(hidden_output, gbow)
            sent_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
            
            word_hidden_output = self.initLinear2(torch.cat((hidden_output, sent_z, gbow),1))
            word_state = torch.cat((word_hidden_output.unsqueeze(0), word_hidden_output.unsqueeze(0)),0)
            output = []
            for k in range(max_dec_len):
                c_t, _ = self.atten(word_hidden_output, video_features, context_mask)
                word_inp = torch.cat((embeddings[:,i,k,:], c_t), 1).unsqueeze(1)
                word_hidden,word_state = self.wordRnn(word_inp,word_state)
                word_hidden_output = word_hidden.squeeze(1)

                output.append(self.wordLinear(word_hidden_output))

            inp = torch.cat((sent_z,word_hidden_output),1).unsqueeze(1)
            
            output = torch.transpose(torch.stack(output), 0, 1) # converting from step_size x batch_size x vocab_size to batch_size x step_size x vocab_size
            target = captions[:,i,:]

            bow_logit = self.bowLinear2(F.tanh(self.bowLinear1(torch.cat((hidden_output, sent_z, gbow),1)))).unsqueeze(1).expand(-1,max_dec_len,-1)
            #if batch_size > 1:
            # Sort by length (keep idx)
            caption_len = caption_lens[:,i]
            caption_len, idx_sort = np.sort(caption_len.cpu().numpy())[::-1], np.argsort(-caption_len.cpu().numpy())

            output = output.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            bow_logit = bow_logit.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            target = target.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            prior_mu =  prior_mu.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            prior_logvar = prior_logvar.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            post_mu = post_mu.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            post_logvar = post_logvar.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            
            if 1 in caption_len:
                num_data = list(caption_len).index(1)
            else:
                num_data = len(caption_len)
            if num_data == 0:
                continue

            caption_len = caption_len[:num_data]
            idx_sort = idx_sort[:num_data]
            output = output[:num_data]
            bow_logit = bow_logit[:num_data]
            target = target[:num_data]

            #print(output.shape,bow_logit.shape,target.shape)
            logit_packed = nn.utils.rnn.pack_padded_sequence(output, caption_len.copy(), batch_first=True)[0]
            bow_logit_packed = nn.utils.rnn.pack_padded_sequence(bow_logit, caption_len.copy(), batch_first=True)[0]
            target_packed = nn.utils.rnn.pack_padded_sequence(target, caption_len.copy(), batch_first=True)[0]
            loss_mask = 1 - caption_lens[:,i].eq(1).float()
            #print(caption_len.shape)
            #print(loss_mask.shape)
            #print(batch_size)
            if i == 0:
                bow_loss = self.ce(bow_logit_packed, target_packed)
                word_loss = self.ce(logit_packed, target_packed)
                local_KL = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar, loss_mask)
                #print('output[0]:',torch.cat(tuple(torch.topk(F.softmax(output[0]),1)[1])))
                #print('target[0]:',captions[0,i,:])
                #print('input[0]:',_captions[0,i,:])
                #print('target:',target)
                #print('target_packed:',target_packed)
                #print('len(target_packed):',len(target_packed))
                #print('len(logit_packed):',len(logit_packed))
            #elif i == 9:
            #    print('caption_len:',caption_len)
            #    print('target_step9:',target)
            #    print('target_packed_step9:',target_packed)
            else:
                bow_loss += self.ce(bow_logit_packed, target_packed)
                word_loss += self.ce(logit_packed, target_packed)
                local_KL += KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar, loss_mask)

        
        return bow_loss, local_KL, word_loss


    def get_prior_latent(self, keywords_encode_t, video_features_t):

        prior_inp = torch.cat((keywords_encode_t, video_features_t), 1)
        prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.args.plan_latent_dim,1)
        return prior_mu , prior_logvar

    def get_post_latent(self, keywords_encode_t, text_encode_t, video_features_t):

        post_inp = torch.cat((keywords_encode_t, video_features_t, text_encode_t), 1)
        post_mu , post_logvar = torch.split(self.post_linear2(F.tanh(self.post_linear1(post_inp))),self.args.plan_latent_dim,1)
        return post_mu , post_logvar

    def beam_search(self, keywords, enc_states, video_features, frame_lens, segments, group_count, \
                    segments_gt, group_count_gt, group_init_state,beam_size):
        batch_size,step_size,_ = segments.shape
        #pdb.set_trace()
        max_len = self.args.max_dec_len
        group_encode = self.GroupEncoder(enc_states, segments, group_count, segments_gt, group_count_gt, 1.0)
        #print(segments,group_count)
        group_encode_t = Variable(torch.zeros(group_encode.shape[0],group_encode.shape[2])).cuda()
        for i in range(batch_size):
            group_encode_t[i] = group_encode[i, group_count[i]-1,:]  # get last state
        
        sent_state = self.initLinear(torch.cat((group_init_state, group_encode_t), 1)).unsqueeze(0)
        sent_inp = Variable(torch.zeros(batch_size,self.decoder_dim+self.sent_latent_dim)).cuda().unsqueeze(1)
        context_mask = rnn_mask(frame_lens, video_features.shape[1])

        output_ids = []
        #output_ids = []
        for ii in range(step_size):

            sent_hidden,sent_state = self.sentRnn(sent_inp,sent_state)
            #print('sent_hidden:',sent_hidden, 'state:',state)
            hidden_output = sent_hidden.squeeze(1)
            segment = segments[:,ii,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            gbow = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
            gbow = self.args.plan_decay * gbow
            #pdb.set_trace()

            prior_mu , prior_logvar = self.get_prior_latent(hidden_output, gbow)
            sent_z = sample_gaussian(prior_mu.shape, prior_mu , prior_logvar)
            
            word_hidden_output = self.initLinear2(torch.cat((hidden_output, sent_z, gbow),1))
            prev_state = torch.cat((word_hidden_output.unsqueeze(0), word_hidden_output.unsqueeze(0)),0)

            embedding = self.embed(Variable(torch.LongTensor(np.ones([batch_size,1]))).cuda()).squeeze(1)

            c_t, _ = self.atten(word_hidden_output, video_features, context_mask)
            
            word_inp = torch.cat((embedding, c_t), 1).unsqueeze(1)
            next_hidden, next_state = self.wordRnn(word_inp,prev_state)
            next_hidden = next_hidden.squeeze(1)

            output = self.wordLinear(next_hidden)
            output = F.softmax(output,1)
            next_probs, next_words = torch.topk(output,beam_size)
            prev_words = torch.t(next_words)
            prev_state = []
            prev_hidden = []

            for i in range(beam_size):
                prev_state.append(next_state)
                prev_hidden.append(next_hidden)
            #print prev_state
            all_probs = next_probs.cpu().data.numpy()

            generated_sequence = np.zeros((batch_size,beam_size,max_len),dtype=np.int32)
            generated_sequence[:,:,0] = next_words.cpu().data.numpy()

            final_results = np.zeros((batch_size,beam_size,max_len), dtype=np.int32)
            final_all_probs = np.zeros((batch_size,beam_size))
            final_results_counter = np.zeros((batch_size),dtype=np.int32) # to check the overflow of beam in fina results


            for i in range(1,max_len):
                probs = []
                state = []
                hidden = []
                words = []

                for j in range(beam_size):
                    inputs = self.embed(prev_words[j])
                    #print inputs
                    c_t, alpha = self.atten(prev_hidden[j], video_features, context_mask)
                    inp = torch.cat((inputs,c_t),1).unsqueeze(1)
                    next_hidden, next_state = self.wordRnn(inp, prev_state[j])
                    next_hidden = next_hidden.squeeze(1)
                    output = self.wordLinear(next_hidden)
                    output = F.softmax(output,1)
                    next_probs, next_words = torch.topk(output, beam_size)
                    probs.append(next_probs)
                    words.append(next_words)
                    state.append(next_state)
                    hidden.append(next_hidden)

                probs = np.transpose(np.array(torch.stack(probs).cpu().data.numpy()),(1,0,2))
                #state = np.transpose(np.array(state.cpu().data.numpy()),(1,0,2))
                hidden = np.transpose(np.array(torch.stack(hidden).cpu().data.numpy()),(1,0,2))
                words = np.transpose(np.array(torch.stack(words).cpu().data.numpy()),(1,0,2))
                #state = [torch.cat(tuple(s),0) for s in state]
                state = torch.stack(state)
                #print state

                prev_state = []
                prev_words = []
                prev_hidden = []

                for k in range(batch_size):
                    probs[k] = np.transpose(np.transpose(probs[k])*all_probs[k]) # multiply each beam words with each beam probs so far
                    top_indices = top_n_indexes(probs[k],beam_size)
                    beam_idx,top_choice_idx = zip(*top_indices)
                    all_probs[k] = (probs[k])[beam_idx,top_choice_idx]
                    prev_state.append([state[idx,:,k,:] for idx in beam_idx])
                    prev_hidden.append([hidden[k,idx,:] for idx in beam_idx])
                    prev_words.append([words[k,idx,idy] for idx,idy in top_indices])
                    generated_sequence[k] = generated_sequence[k,beam_idx,:]
                    generated_sequence[k,:,i] = [words[k,idx,idy] for idx,idy in top_indices]



                    # code to extract complete summaries ending with [EOS] or [STOP] or [END]

                    for beam_idx in range(beam_size):
                        if generated_sequence[k,beam_idx,i] == 2 and final_results_counter[k]<beam_size: # [EOS] or [STOP] or [END] word / check overflow
                            # print generated_sequence[k,beam_idx]
                            final_results[k,final_results_counter[k],:] = generated_sequence[k,beam_idx,:]
                            final_all_probs[k,final_results_counter[k]] = all_probs[k,beam_idx]
                            final_results_counter[k] += 1 
                            all_probs[k,beam_idx] = 0.0 # supress this sentence to flow further through the beam


                if np.sum(final_results_counter) == batch_size*beam_size: # when suffiecient hypothsis are obtained i.e. beam size hypotheis, break the process
                    # print "Encounter a case"
                    break

                # transpose batch to usual
                #print prev_state
                prev_state = [torch.stack(s,0) for s in prev_state]
                prev_state = torch.stack(prev_state,0)
                prev_state = torch.transpose(prev_state,0,1)
                tmp_state = torch.transpose(prev_state,1,2)
                prev_state = []
                for k in range(beam_size):
                    prev_state.append(tmp_state[k,:,:,:].contiguous())

                #print prev_state
                prev_words = np.transpose(np.array(prev_words),(1,0)) # set order [beam_size, batch_size]
                prev_words = Variable(torch.LongTensor(prev_words)).cuda()
                prev_hidden = np.transpose(np.array(prev_hidden),(1,0,2))
                prev_hidden = Variable(torch.FloatTensor(prev_hidden)).cuda()
                #print prev_hidden[0]
                #print prev_state[0]
                #print generated_sequence
                


            sampled_ids = []
            for k in range(batch_size):
                avg_log_probs = []
                for j in range(beam_size):
                    try:
                        num_tokens = final_results[k,j,:].tolist().index(2)+1 #find the stop word and get the lenth of the sequence based on that
                    except:
                        num_tokens = 1 # this case is when the number of hypotheis are not equal to beam size, i.e., durining the process sufficinet hypotheisis are not obtained
                    if num_tokens == 0:
                        num_tokens = 1
                    probs = np.where(final_all_probs[k][j]!=0, np.log(final_all_probs[k][j]) ,0)

                    avg_log_probs.append(probs)
                avg_log_probs = np.array(avg_log_probs)
                sort_order = np.argsort(avg_log_probs)
                sort_order[:] = sort_order[::-1]
                sort_generated_sequence  = final_results[k,sort_order,:]
                sampled_ids.append(sort_generated_sequence[0])
            output_ids.append(sampled_ids)
            #print('output_ids:', output_ids)
            #print(sampled_ids)
            sent_inp = torch.cat((sent_z,next_hidden),1).unsqueeze(1)
        output_ids = [list(row) for row in zip(*output_ids)]  # 转置
        #print(output_ids)
        output_ids = self._agg_group(group_count, output_ids)
        plans = self._agg_group_plan(group_count,segments, keywords)
        gt_plans = self._agg_group_plan(group_count_gt,segments_gt, keywords)
        return plans, gt_plans, output_ids


    def _agg_group(self, stop, text):

        translation = []
        for gcnt, sent in zip(stop, text):
            sent = sent[:gcnt]
            desc = []
            for segId, seg in enumerate(sent):
                for wid in seg:
                    if wid == 2:  #end_token
                        desc.append(wid)
                        break
                    elif wid == 0 or wid == 1:  # start_token or pad
                        continue
                    else:
                        desc.append(wid)
            translation.append(desc)

        return translation
    
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



class Attention(nn.Module):
    def __init__(self, args, enc_dim, dec_dim, attn_dim=None):
        super(Attention, self).__init__()
        
        self.args = args
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = self.dec_dim if attn_dim is None else attn_dim


        self.encoder_in = nn.Linear(self.enc_dim, self.attn_dim, bias=True)
        self.decoder_in = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.attn_linear = nn.Linear(self.attn_dim, 1, bias=False)


    def forward(self, dec_state, enc_states, mask, dag=None):
        """
        :param dec_state: 
            decoder hidden state of size batch_size x dec_dim
        :param enc_states:
            all encoder hidden states of size batch_size x max_enc_steps x vid_hid
        :param flengths:
            encoder video frame lengths of size batch_size
        """
        dec_contrib = self.decoder_in(dec_state)
        batch_size, max_enc_steps, _  = enc_states.size()
        enc_contrib = self.encoder_in(enc_states.contiguous().view(-1, self.enc_dim)).contiguous().view(batch_size, max_enc_steps, self.attn_dim)
        pre_attn = F.tanh(enc_contrib + dec_contrib.unsqueeze(1).expand_as(enc_contrib))
       
        
        energy = self.attn_linear(pre_attn.view(-1, self.attn_dim)).view(batch_size, max_enc_steps)
        alpha = F.softmax(energy, 1)
        # mask alpha and renormalize it
        alpha = alpha* mask
        alpha = torch.div(alpha, alpha.sum(1).unsqueeze(1).expand_as(alpha))

        context_vector = torch.bmm(alpha.unsqueeze(1), enc_states).squeeze(1) # (batch_size, vid_hid)

        return context_vector, alpha

class PHVM_Transformer(nn.Module):
    def __init__(self, args):
        super(PHVM_Transformer, self).__init__()
        self.args = args

        self.FramesEncoder = FramesEncoder_Trans(self.args)
        self.KeywordsEncoder = WordsEncoder_Trans(self.args)
        self.TextsEncoder = WordsEncoder_Trans(self.args)
        self.GroupDecoder = GroupDecoder_Trans(self.args)
        self.Decoder = Decoder_Trans(self.args)

        self.prior_linear1 = nn.Linear(self.args.model_dim*2, self.args.plan_latent_dim*2)
        self.prior_linear2 = nn.Linear(self.args.plan_latent_dim*2, self.args.plan_latent_dim*2)
        self.post_linear1 = nn.Linear(self.args.model_dim*3, self.args.plan_latent_dim*2)
        self.post_linear2 = nn.Linear(self.args.plan_latent_dim*2, self.args.plan_latent_dim*2)


    def forward(self, frames, frame_lens, keywords, keyword_lens, texts, texts_lens, segments, group_count, captions, caption_lens):

        video_features = self.FramesEncoder(frames, frame_lens)
        video_features_t = torch.sum(video_features,1)
        
        keywords_encode = self.KeywordsEncoder(keywords,keyword_lens)
        keywords_encode_t = torch.sum(keywords_encode,1) 

        text_encode = self.TextsEncoder(texts,texts_lens)
        text_encode_t = torch.sum(text_encode,1) 

        post_mu , post_logvar = self.get_post_latent(keywords_encode_t, text_encode_t, video_features_t)
        prior_mu , prior_logvar = self.get_prior_latent(keywords_encode_t, video_features_t)
        global_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
        
        group_init_state = torch.cat((keywords_encode_t, global_z), 1)
        memory =  torch.cat((keywords_encode, global_z.unsqueeze(1).expand(-1,keywords_encode.shape[1],-1)), 2)
        stop_loss, group_loss = self.GroupDecoder(keywords.transpose(0,1), keywords_encode, segments, group_count, memory, keyword_lens)

        bow_loss, local_KL, word_loss = self.Decoder(frames.transpose(0,1), keywords_encode, video_features, frame_lens, segments, group_count, captions, caption_lens, group_init_state)


        global_KL = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)

        #loss = stop_loss + word_loss + group_loss + global_KL + local_KL + bow_loss


        return stop_loss , word_loss , group_loss , global_KL , local_KL , bow_loss


    def get_prior_latent(self, keywords_encode_t, video_features_t):

        prior_inp = torch.cat((keywords_encode_t, video_features_t), 1)
        prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.args.plan_latent_dim,1)
        return prior_mu , prior_logvar

    def get_post_latent(self, keywords_encode_t, text_encode_t, video_features_t):

        post_inp = torch.cat((keywords_encode_t, video_features_t, text_encode_t), 1)
        post_mu , post_logvar = torch.split(self.post_linear2(F.tanh(self.post_linear1(post_inp))),self.args.plan_latent_dim,1)
        return post_mu , post_logvar


    def beam_search(self, frames, frame_lens, keywords, keyword_lens, segments_gt, group_count_gt):

        video_features = self.FramesEncoder(frames, frame_lens)
        video_features_t = torch.sum(video_features,1)
        
        keywords_encode = self.KeywordsEncoder(keywords,keyword_lens)
        keywords_encode_t = torch.sum(keywords_encode,1) 

        prior_mu , prior_logvar = self.get_prior_latent(keywords_encode_t, video_features_t)
        global_z = sample_gaussian(prior_mu.shape, prior_mu , prior_logvar)
        
        group_init_state = torch.cat((keywords_encode_t, global_z), 1)
        memory =  torch.cat((keywords_encode, global_z.unsqueeze(1).expand(-1,keywords_encode.shape[1],-1)), 2)

        if not self.args.use_gt_plan:
            segments_p, group_count_p = self.GroupDecoder.sample(keywords.transpose(0,1), keywords_encode, keyword_lens, memory)
        else:
            segments_p = segments_gt
            group_count_p =  group_count_gt
        used_plans, gt_plans, output_ids = self.Decoder.beam_search(frames.transpose(0,1), keywords, keywords_encode, video_features, \
        frame_lens, segments_p, group_count_p, segments_gt, group_count_gt,  group_init_state, self.args.beam_size)

        return used_plans, gt_plans, output_ids  



class FramesEncoder_Trans(nn.Module):
    def __init__(self, args):
        super(FramesEncoder_Trans, self).__init__()

        self.args = args
        self.embed_size = self.args.embed_size
        self.vid_dim = self.args.vid_dim
        self.model_dim = self.args.model_dim
        self.layer = self.args.video_trans_layer
        self.heads = self.args.video_heads
        self.enable_cuda = self.args.cuda

        self.vecembed = VecEmbedding(self.vid_dim, self.model_dim, position_encoding=self.args.position_encoding)
        self.trans = TransformerEncoder(self.layer,self.model_dim,self.heads,self.args.transformer_ff,self.args.dropout,self.args.attention_dropout,self.vecembed,self.args.max_relative_positions)
        

    def forward(self, frames, flengths):
        """Handles variable size frames
           frame_embed: video features(B,T,F)
           flengths: frame lengths
        """
        #pdb.set_trace()
        _, outputs, _ = self.trans(frames.transpose(0,1),flengths)
        return outputs.transpose(0,1)

# Based on tutorials/08 - Language Model
# RNN Based Language Model
class WordsEncoder_Trans(nn.Module):
    def __init__(self, args):
        super(WordsEncoder_Trans, self).__init__()

        self.args = args
        self.embed_size = self.args.embed_size
        self.vocab_size = self.args.vocab_size
        self.model_dim = self.args.model_dim
        self.layer = self.args.encoder_trans_layer
        self.heads = self.args.encoder_heads
        self.enable_cuda = self.args.cuda

        self.vecembed = VecEmbedding(self.embed_size, self.model_dim, position_encoding=self.args.position_encoding)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.trans = TransformerEncoder(self.layer,self.model_dim,self.heads,self.args.transformer_ff,self.args.dropout,self.args.attention_dropout,self.vecembed,self.args.max_relative_positions)

    def forward(self, inputs, lengths):
        """Handles variable size inputs
           frame_embed: video features
           flengths: frame lengths
        """

        embeddings = self.embed(inputs).transpose(0,1)
        _, outputs, _ = self.trans(embeddings,lengths)

        return outputs.transpose(0,1)

class GroupEncoder_Trans(nn.Module):
    def __init__(self, args):
        super(GroupEncoder_Trans, self).__init__()

        # self.use_abs = use_abs
        self.args = args

        
        self.model_dim = self.args.model_dim
        self.max_group_count = self.args.max_group_count
        self.plan_latent_dim = self.args.plan_latent_dim
        self.layer = self.args.group_decoder_trans_layer
        self.heads = self.args.group_decoder_heads

        self.dropout_rate = self.args.dropout

        self.enable_cuda = self.args.cuda
        
        self.Linear1 = nn.Linear(self.model_dim+self.plan_latent_dim, self.model_dim,bias=False)

        self.vecembed = VecEmbedding(self.model_dim*2, self.model_dim, position_encoding=self.args.position_encoding)
        self.trans = TransformerEncoder(self.layer,self.model_dim,self.heads,self.args.transformer_ff,self.args.dropout,self.args.attention_dropout,self.vecembed,self.args.max_relative_positions)

    def forward(self, enc_states, segments, group_count, group_init_state):
        batch_size,step_size,_ = segments.shape

        segments_input = torch.cat((Variable(torch.LongTensor(np.zeros([batch_size,1,segments.shape[2]]))).cuda(),segments), 1) #key start group
        inps = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,self.model_dim]))).cuda()
        for i in range(step_size):
            segment = segments_input[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inps[:,i,:] = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)


        inps = torch.cat((inps.transpose(0,1), self.Linear1(group_init_state.unsqueeze(0).expand(step_size,-1,-1))), 2)

        _, outputs, _ = self.trans(inps,group_count)

        return outputs.transpose(0,1)


class GroupDecoder_Trans(nn.Module):
    def __init__(self, args):
        super(GroupDecoder_Trans, self).__init__()

        # self.use_abs = use_abs
        self.args = args

        
        self.model_dim = self.args.model_dim
        self.max_group_count = self.args.max_group_count
        self.plan_latent_dim = self.args.plan_latent_dim
        self.layer = self.args.group_decoder_trans_layer
        self.heads = self.args.group_decoder_heads


        self.enable_cuda = self.args.cuda
        
        self.vecembed = VecEmbedding(self.model_dim, self.model_dim, position_encoding=self.args.position_encoding)
        self.Linear1 = nn.Linear(self.model_dim+self.plan_latent_dim, self.model_dim,bias=False)
        self.trans = TransformerDecoder(self.layer,self.model_dim,self.heads,self.args.transformer_ff,False,'scaled-dot', \
                                        self.args.dropout,self.args.attention_dropout,self.vecembed,self.args.max_relative_positions,True,True,-3,0)
        

        self.PlanLinear1 = nn.Linear(self.model_dim*2, self.model_dim)
        self.PlanLinear2 = nn.Linear(self.model_dim, 1)
        self.StopLinear = nn.Linear(self.model_dim, 1)
        
        

    def forward(self, keywords, enc_states, segments, group_count, memory, keyword_lens):

        batch_size,step_size,_ = segments.shape


        segments_input = torch.cat((Variable(torch.LongTensor(np.zeros([batch_size,1,segments.shape[2]]))).cuda(),segments), 1) #key start group
        inps = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,self.model_dim]))).cuda()
        for i in range(step_size):
            segment = segments_input[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inps[:,i,:] = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
        inps = inps.transpose(0,1)

        memory = self.Linear1(memory).transpose(0,1)

        self.trans.init_state(keywords, memory, memory)
        #pdb.set_trace()
        #print(keyword_lens)
        outputs, _ = self.trans(inps, memory, memory_lengths=keyword_lens)
        outputs = outputs.transpose(0,1) #转回batch_first

        group_count, idx_sort = np.sort(group_count.cpu().numpy())[::-1], np.argsort(-group_count.cpu().numpy())
        #pdb.set_trace()
        outputs = outputs.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
        outputs = nn.utils.rnn.pack_padded_sequence(outputs, group_count.copy(), batch_first=True)[0]
        group_targets_packed = nn.utils.rnn.pack_padded_sequence(segments, group_count.copy(), batch_first=True)[0]
    
        #key data_len * max_enc_len
        enc_state_2 = nn.utils.rnn.pack_padded_sequence(enc_states.unsqueeze(1).expand(-1,step_size,-1,-1), group_count.copy(), batch_first=True)[0] #key packed_data_len * max_enc_len * enc_dim
        enc_state_2 = torch.cat((enc_state_2, Variable(torch.FloatTensor(np.zeros([enc_state_2.shape[0],segments.shape[2]-enc_state_2.shape[1],self.model_dim]))).cuda()),1)
        outputs_2 = outputs.unsqueeze(1).expand(-1,segments.shape[2],-1) #key packed_data_len * max_enc_len * hidden_dim
        outputs_2 = torch.cat((outputs_2, enc_state_2),2)

        stop_label = Variable(torch.zeros(batch_size, step_size)).cuda()
        for i in range(batch_size):
            stop_label[i,group_count[i] -1] = 1

        stop_packed = nn.utils.rnn.pack_padded_sequence(stop_label, group_count.copy(), batch_first=True)[0]

        group_loss = F.binary_cross_entropy_with_logits(self.PlanLinear2(F.tanh(self.PlanLinear1(outputs_2))).squeeze(2), group_targets_packed.float())
        stop_loss = F.binary_cross_entropy_with_logits(self.StopLinear(outputs).squeeze(1), stop_packed.float()) 
        return stop_loss, group_loss

    def sample(keywords, enc_states, keyword_lens, memory):
        batch_size,max_enc_len,_ = enc_states.shape
        step_size = self.args.max_group_count

        inp = Variable(torch.FloatTensor(np.zeros([batch_size,1,self.model_dim]))).cuda()
        stop_pred = Variable(torch.FloatTensor(np.zeros([batch_size,step_size]))).cuda()
        segments = Variable(torch.LongTensor(np.zeros([batch_size,step_size,max_enc_len]))).cuda()
        group_count = torch.LongTensor(np.zeros(batch_size))

        self.trans.init_state(keywords, memory, memory)
        for i in range(step_size):

            memory = self.Linear1(memory).transpose(0,1)
            outputs, _ = self.trans(inp.transpose(0,1), memory, memory_lengths=keyword_lens, step=i)
            outputs = outputs.transpose(0,1) #转回batch_first

            enc_state_2 = torch.cat((enc_states, Variable(torch.FloatTensor(np.zeros([enc_states.shape[0],max_enc_len-enc_states.shape[1],self.encoder_dim*2]))).cuda()),1)
            outputs_2 = outputs.expand(-1,max_enc_len,-1) #key batch_size * max_enc_len * hidden_dim
            outputs_2 = torch.cat((outputs_2, enc_state_2),2)

            plan_pred = F.sigmoid(self.PlanLinear2(F.tanh(self.PlanLinear1(outputs_2))).squeeze(2))  # batch_size*max_enc_len
            stop_pred[:,i] = F.sigmoid(self.StopLinear(outputs.squeeze(1)).squeeze(1)) # batch*step
            
            #pdb.set_trace()
            segment = Variable((plan_pred > 0.5).long()).cuda()
            for j in range(batch_size):
                if torch.sum(segment[j]) == 0:
                    segment[j][torch.argmax(plan_pred[j])] = 1
            
            segments[:,i,:] = segment
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inp = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states) # (batch_size, 1, enc_dim)
        
        for j in range(batch_size):
            if len((stop_pred[j]>0.5).nonzero()) != 0:
                group_count[j] = (stop_pred[j]>0.5).nonzero()[0] + 1 #第一个大于0.5的值的索引 + 1
            else:
                group_count[j] = torch.argmax(stop_pred[j]) + 1
        #print('group_count:',group_count)

        return segments, group_count  # batch_size*max_step*max_keyword_len, array (batch_size*1)





class Decoder_Trans(nn.Module):
    def __init__(self, args):
        super(Decoder_Trans, self).__init__()

        # self.use_abs = use_abs
        self.args = args
        self.model_dim = self.args.model_dim
        self.sent_decoder_trans_layer = self.args.sent_decoder_trans_layer
        self.sent_decoder_heads = self.args.sent_decoder_heads
        self.decoder_trans_layer = self.args.decoder_trans_layer
        self.decoder_heads = self.args.decoder_heads
        
        
        self.sent_latent_dim = self.args.sent_latent_dim

        self.bow_hid = self.args.bow_hid
        self.plan_latent_dim = self.args.plan_latent_dim

        self.embed_size = self.args.embed_size
        self.vocab_size = self.args.vocab_size

        self.enable_cuda = self.args.cuda
        self.dropout_rate = self.args.dropout

        self.GroupEncoder = GroupEncoder_Trans(self.args)

        self.SentEncoder = WordsEncoder_Trans(self.args)

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.vecembed1 = VecEmbedding(self.model_dim*2+self.sent_latent_dim, self.model_dim, position_encoding=self.args.position_encoding)
        self.vecembed2 = VecEmbedding(self.model_dim*2, self.model_dim, position_encoding=self.args.position_encoding)
        
        
        self.sent_trans = TransformerEncoder(self.sent_decoder_trans_layer,self.model_dim,self.sent_decoder_heads, self.args.transformer_ff,\
                                            self.args.dropout,self.args.attention_dropout,self.vecembed1,self.args.max_relative_positions)

        self.trans = TransformerDecoder(self.decoder_trans_layer,self.model_dim,self.decoder_heads,self.args.transformer_ff,False,'scaled-dot', \
                                        self.args.dropout,self.args.attention_dropout,self.vecembed2,self.args.max_relative_positions,True,True,-3,0)

        self.bowLinear1 = nn.Linear(self.args.sent_latent_dim+self.model_dim*2, self.bow_hid)
        self.bowLinear2 = nn.Linear(self.bow_hid, self.vocab_size)
        self.wordLinear = nn.Linear(self.model_dim, self.vocab_size)
        self.Linear0 = nn.Linear(self.embed_size, self.model_dim)
        self.Linear2 = nn.Linear(self.model_dim*2+self.sent_latent_dim, self.model_dim)

        self.prior_linear1 = nn.Linear(self.args.model_dim*2, self.args.sent_latent_dim*2)
        self.prior_linear2 = nn.Linear(self.args.sent_latent_dim*2, self.args.sent_latent_dim*2)
        self.post_linear1 = nn.Linear(self.args.model_dim*3, self.args.sent_latent_dim*2)
        self.post_linear2 = nn.Linear(self.args.sent_latent_dim*2, self.args.sent_latent_dim*2)

        self.ce = nn.CrossEntropyLoss()
        

    def forward(self, frames, enc_states, video_features, frame_lens, segments, group_count, captions, caption_lens, group_init_state):

        batch_size,step_size,max_caption_len = captions.shape

        group_encode = self.GroupEncoder(enc_states, segments, group_count, group_init_state)
        group_encode_t = torch.sum(group_encode,1)

        inp = torch.cat((group_init_state, group_encode_t),1).unsqueeze(0)
        _captions = torch.cat((Variable(torch.LongTensor(np.ones([batch_size,step_size,1]))).cuda(),captions), 2)[:,:,:-1]
        embeddings = self.embed(_captions)

        self.trans.init_state(frames, frames, frames)

        for i in range(step_size):
            #if i==2:
            #    pdb.set_trace()
            #print('inp.shape',inp.shape)
            lengths = torch.ones(batch_size).cuda()
            _, sent_out, _ = self.sent_trans(inp,lengths)
            sent_out = sent_out.squeeze(0)

            segment = segments[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            gbow = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)
            
            sent_encode = self.SentEncoder(captions[:,i,:caption_lens[:,i].max()],caption_lens[:,i])
            sent_encode_t = torch.sum(sent_encode,1)
            post_mu , post_logvar = self.get_post_latent(sent_out, sent_encode_t, gbow)
            prior_mu , prior_logvar = self.get_prior_latent(sent_out, gbow)
            sent_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
            
            #for k in range(max_caption_len):
            init_states = self.Linear2(torch.cat((sent_z,sent_out,gbow),1)).unsqueeze(1).expand(-1,max_caption_len,-1)
            word_inp = torch.cat((self.Linear0(embeddings[:,i,:,:]), init_states), 2).transpose(0,1)
            output, _ = self.trans(word_inp,video_features.transpose(0,1),memory_lengths=frame_lens)
            output = output.transpose(0,1)

            inp = torch.cat((sent_z,torch.sum(output,1),gbow),1).unsqueeze(0)

            output = self.wordLinear(output)
            
            target = captions[:,i,:]

            bow_logit = self.bowLinear2(F.tanh(self.bowLinear1(torch.cat((sent_z, sent_out, gbow),1)))).unsqueeze(1).expand(-1,max_caption_len,-1)
            #if batch_size > 1:
            # Sort by length (keep idx)
            caption_len = caption_lens[:,i]
            caption_len, idx_sort = np.sort(caption_len.cpu().numpy())[::-1], np.argsort(-caption_len.cpu().numpy())

            output = output.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            bow_logit = bow_logit.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            target = target.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            prior_mu =  prior_mu.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            prior_logvar = prior_logvar.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            post_mu = post_mu.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
            post_logvar = post_logvar.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))

            if 1 in caption_len:
                num_data = list(caption_len).index(1)
            else:
                num_data = len(caption_len)
            if num_data == 0:
                continue

            caption_len = caption_len[:num_data]
            idx_sort = idx_sort[:num_data]
            output = output[:num_data]
            bow_logit = bow_logit[:num_data]
            target = target[:num_data]

            prior_mu =  prior_mu[:num_data]
            prior_logvar = prior_logvar[:num_data]
            post_mu = post_mu[:num_data]
            post_logvar = post_logvar[:num_data]

            #print(output.shape,bow_logit.shape,target.shape)
            logit_packed = nn.utils.rnn.pack_padded_sequence(output, caption_len.copy(), batch_first=True)[0]
            bow_logit_packed = nn.utils.rnn.pack_padded_sequence(bow_logit, caption_len.copy(), batch_first=True)[0]
            target_packed = nn.utils.rnn.pack_padded_sequence(target, caption_len.copy(), batch_first=True)[0]
            #print(caption_len.shape)
            #print(loss_mask.shape)
            #print(batch_size)
            if i == 0:
                bow_loss = self.ce(bow_logit_packed, target_packed)
                word_loss = self.ce(logit_packed, target_packed)
                local_KL = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)
                #print('output[0]:',torch.cat(tuple(torch.topk(F.softmax(output[0]),1)[1])))
                #print('target[0]:',captions[0,i,:])
                #print('input[0]:',_captions[0,i,:])
                #print('target:',target)
                #print('target_packed:',target_packed)
                #print('len(target_packed):',len(target_packed))
                #print('len(logit_packed):',len(logit_packed))
            #elif i == 9:
            #    print('caption_len:',caption_len)
            #    print('target_step9:',target)
            #    print('target_packed_step9:',target_packed)
            else:
                bow_loss += self.ce(bow_logit_packed, target_packed)
                word_loss += self.ce(logit_packed, target_packed)
                local_KL += KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)

        return bow_loss, local_KL, word_loss


    def get_prior_latent(self, keywords_encode_t, video_features_t):

        prior_inp = torch.cat((keywords_encode_t, video_features_t), 1)
        prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.args.plan_latent_dim,1)
        return prior_mu , prior_logvar

    def get_post_latent(self, keywords_encode_t, text_encode_t, video_features_t):

        post_inp = torch.cat((keywords_encode_t, video_features_t, text_encode_t), 1)
        post_mu , post_logvar = torch.split(self.post_linear2(F.tanh(self.post_linear1(post_inp))),self.args.plan_latent_dim,1)
        return post_mu , post_logvar

    def beam_search(self,frames, keywords, enc_states, video_features, frame_lens, segments, group_count, \
                    segments_gt, group_count_gt, group_init_state, beam_size):
        batch_size,step_size,_ = segments.shape
        max_len = self.args.max_dec_len
        
        group_encode = self.GroupEncoder(enc_states, segments, group_count, group_init_state)
        group_encode_t = torch.sum(group_encode,1)

        sent_inp = torch.cat((group_init_state, group_encode_t),1).unsqueeze(0)
        self.trans.init_state(frames, frames, frames)
        
        output_ids = []
        #output_ids = []

        for ii in range(step_size):
            lengths = torch.ones(batch_size).cuda()
            _, sent_out, _ = self.sent_trans(sent_inp,lengths)
            sent_out = sent_out.squeeze(0)

            segment = segments[:,ii,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            gbow = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)

            prior_mu , prior_logvar = self.get_prior_latent(sent_out, gbow)
            sent_z = sample_gaussian(prior_mu.shape, prior_mu , prior_logvar)
            
            embedding = self.embed(Variable(torch.LongTensor(np.ones([batch_size,1]))).cuda()) #start token
            
            init_states = self.Linear2(torch.cat((sent_z,sent_out,gbow),1)).unsqueeze(1)
            word_inp = torch.cat((self.Linear0(embedding), init_states), 2).transpose(0,1)
            next_hidden, _ = self.trans(word_inp,video_features.transpose(0,1),memory_lengths=frame_lens,step=ii)
            next_hidden = next_hidden.squeeze(0)

            output = self.wordLinear(next_hidden)
            output = F.softmax(output,1)
            next_probs, next_words = torch.topk(output,beam_size)
            prev_words = torch.t(next_words)
            prev_hidden = []

            for i in range(beam_size):
                prev_hidden.append(next_hidden)
            all_probs = next_probs.cpu().data.numpy()

            generated_sequence = np.zeros((batch_size,beam_size,max_len),dtype=np.int32)
            generated_sequence[:,:,0] = next_words.cpu().data.numpy()

            final_results = np.zeros((batch_size,beam_size,max_len), dtype=np.int32)
            final_all_probs = np.zeros((batch_size,beam_size))
            final_results_counter = np.zeros((batch_size),dtype=np.int32) # to check the overflow of beam in fina results


            for i in range(1,max_len):
                #if i==8:
                #    pdb.set_trace()
                probs = []
                state = []
                hidden = []
                words = []

                for j in range(beam_size):
                    inputs = self.embed(prev_words[j]).unsqueeze(1)
                    word_inp = torch.cat((self.Linear0(inputs), init_states), 2).transpose(0,1)
                    next_hidden, _ = self.trans(word_inp,video_features.transpose(0,1),memory_lengths=frame_lens,step=ii)
                    next_hidden = next_hidden.squeeze(0)
                    output = self.wordLinear(next_hidden)
                    output = F.softmax(output,1)
                    next_probs, next_words = torch.topk(output, beam_size)
                    probs.append(next_probs)
                    words.append(next_words)
                    hidden.append(next_hidden)

                probs = np.transpose(np.array(torch.stack(probs).cpu().data.numpy()),(1,0,2))
                #state = np.transpose(np.array(state.cpu().data.numpy()),(1,0,2))
                hidden = np.transpose(np.array(torch.stack(hidden).cpu().data.numpy()),(1,0,2))
                words = np.transpose(np.array(torch.stack(words).cpu().data.numpy()),(1,0,2))

                prev_words = []
                prev_hidden = []

                for k in range(batch_size):
                    probs[k] = np.transpose(np.transpose(probs[k])*all_probs[k]) # multiply each beam words with each beam probs so far
                    top_indices = top_n_indexes(probs[k],beam_size)
                    beam_idx,top_choice_idx = zip(*top_indices)
                    all_probs[k] = (probs[k])[beam_idx,top_choice_idx]
                    prev_hidden.append([hidden[k,idx,:] for idx in beam_idx])
                    prev_words.append([words[k,idx,idy] for idx,idy in top_indices])
                    generated_sequence[k] = generated_sequence[k,beam_idx,:]
                    generated_sequence[k,:,i] = [words[k,idx,idy] for idx,idy in top_indices]



                    # code to extract complete summaries ending with [EOS] or [STOP] or [END]

                    for beam_idx in range(beam_size):
                        if generated_sequence[k,beam_idx,i] == 2 and final_results_counter[k]<beam_size: # [EOS] or [STOP] or [END] word / check overflow
                            # print generated_sequence[k,beam_idx]
                            final_results[k,final_results_counter[k],:] = generated_sequence[k,beam_idx,:]
                            final_all_probs[k,final_results_counter[k]] = all_probs[k,beam_idx]
                            final_results_counter[k] += 1 
                            all_probs[k,beam_idx] = 0.0 # supress this sentence to flow further through the beam


                if np.sum(final_results_counter) == batch_size*beam_size: # when suffiecient hypothsis are obtained i.e. beam size hypotheis, break the process
                    # print "Encounter a case"
                    break


                prev_words = np.transpose(np.array(prev_words),(1,0)) # set order [beam_size, batch_size]
                prev_words = Variable(torch.LongTensor(prev_words)).cuda()
                prev_hidden = np.transpose(np.array(prev_hidden),(1,0,2))
                prev_hidden = Variable(torch.FloatTensor(prev_hidden)).cuda()
                #print prev_hidden[0]
                #print generated_sequence
                


            sampled_ids = []
            for k in range(batch_size):
                avg_log_probs = []
                for j in range(beam_size):
                    try:
                        num_tokens = final_results[k,j,:].tolist().index(2)+1 #find the stop word and get the lenth of the sequence based on that
                    except:
                        num_tokens = 1 # this case is when the number of hypotheis are not equal to beam size, i.e., durining the process sufficinet hypotheisis are not obtained
                    if num_tokens == 0:
                        num_tokens = 1
                    probs = np.where(final_all_probs[k][j]!=0, np.log(final_all_probs[k][j]) ,0)

                    avg_log_probs.append(probs)
                avg_log_probs = np.array(avg_log_probs)
                sort_order = np.argsort(avg_log_probs)
                sort_order[:] = sort_order[::-1]
                sort_generated_sequence  = final_results[k,sort_order,:]
                sampled_ids.append(sort_generated_sequence[0])
            output_ids.append(sampled_ids)
            #print('output_ids:', output_ids)
            #print(sampled_ids)
            sent_inp = torch.cat((sent_z,next_hidden,group_encode_t),1).unsqueeze(0)
        output_ids = [list(row) for row in zip(*output_ids)]  # 转置
        output_ids = self._agg_group(group_count, output_ids)
        plans = self._agg_group_plan(group_count,segments, keywords)
        gt_plans = self._agg_group_plan(group_count_gt,segments_gt, keywords)
        return plans, gt_plans, output_ids


    def _agg_group(self, stop, text):

        translation = []
        for gcnt, sent in zip(stop, text):
            sent = sent[:gcnt]
            desc = []
            for segId, seg in enumerate(sent):
                for wid in seg:
                    if wid == 2:  #end_token
                        desc.append(wid)
                        break
                    elif wid == 0 or wid == 1:  # start_token or pad
                        continue
                    else:
                        desc.append(wid)
            translation.append(desc)

        return translation
    
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
