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
import bottleneck as bn
import pdb

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


class PHVMConfig:
    def __init__(self):
        super(PHVMConfig, self).__init__()
        #cuda
        self.cuda = True
        
        # rnn
        self.birnn = True
        self.PHVM_rnn_type = 'gru'

        #transformer
        self.model_dim = 1024

        #video
        self.video_dim = 2048
        self.video_hid = 1024
        self.video_layer = 4
        self.video_heads = 8

        # embedding
        self.vocab_size = 4118
        self.share_vocab = False
        self.PHVM_word_dim = 512
        self.PHVM_key_dim = 30
        self.PHVM_val_dim = 100
        self.PHVM_cate_dim = 10

        # group
        self.PHVM_group_selection_threshold = 0.5
        self.PHVM_stop_threshold = 0.5
        self.PHVM_max_group_cnt = 30
        self.PHVM_max_sent_cnt = 10

        # type
        self.PHVM_use_type_info = False
        self.PHVM_type_dim = 30

        # encoder
        self.max_group_count = 10
        self.max_enc_len = 35
        self.PHVM_encoder_dim = 100
        self.encoder_layer = 4
        self.encoder_heads = 8

        # group_decoder
        self.PHVM_group_decoder_dim = 100
        self.group_decoder_layer = 1
        self.group_decoder_heads = 8


        # group encoder
        self.group_encoder_layer = 1
        self.group_encoder_heads = 8

        # sent_decoder
        self.sent_decoder_layer = 1
        self.sent_decoder_heads = 8

        # sent_top_encoder
        self.PHVM_sent_top_encoder_dim = 300
        self.PHVM_sent_top_encoder_num_layer = 1

        # text post encoder
        self.PHVM_text_post_encoder_dim = 300
        self.PHVM_text_post_encoder_num_layer = 1

        # sent_post_encoder
        self.PHVM_sent_post_encoder_dim = 300
        self.PHVM_sent_post_encoder_num_layer = 1

        # bow
        self.PHVM_bow_hidden_dim = 200

        # decoder
        self.decoder_layer = 4
        self.decoder_heads = 8
        self.max_caption_len = 30

        # latent
        self.PHVM_plan_latent_dim = 512
        self.PHVM_sent_latent_dim = 512

        # training
        self.PHVM_learning_rate = 0.001
        self.PHVM_num_training_step = 100000
        self.PHVM_sent_full_KL_step = 20000
        self.PHVM_plan_full_KL_step = 40000
        self.PHVM_dropout = 0.4 

        # inference
        self.PHVM_beam_width = 10
        self.PHVM_maximum_iterations = 50

class PHVM(nn.Module):
    def __init__(self, args, config=PHVMConfig()):
        super(PHVM, self).__init__()
        self.config = config
        self.early_stopping = 15

        self.birnn = config.birnn
        self.FramesEncoder = FramesEncoder(config)
        self.KeywordsEncoder = WordsEncoder(config)
        self.TextsEncoder = WordsEncoder(config)
        self.GroupDecoder = GroupDecoder(config)
        self.Decoder = Decoder(config)

        self.prior_linear1 = nn.Linear(config.model_dim*2, config.PHVM_plan_latent_dim*2)
        self.prior_linear2 = nn.Linear(config.PHVM_plan_latent_dim*2, config.PHVM_plan_latent_dim*2)
        self.post_linear1 = nn.Linear(config.model_dim*3, config.PHVM_plan_latent_dim*2)
        self.post_linear2 = nn.Linear(config.PHVM_plan_latent_dim*2, config.PHVM_plan_latent_dim*2)


        self.prior_linear1.weight.data.uniform_(-0.08, 0.08)
        self.prior_linear1.bias.data.fill_(0)
        self.prior_linear2.weight.data.uniform_(-0.08, 0.08)
        self.prior_linear2.bias.data.fill_(0)
        self.post_linear1.weight.data.uniform_(-0.08, 0.08)
        self.post_linear1.bias.data.fill_(0)
        self.post_linear2.weight.data.uniform_(-0.08, 0.08)
        self.post_linear2.bias.data.fill_(0)

        

    def forward(self, frames, frame_lens, keywords, keyword_lens, texts, texts_lens, segments, group_count, group_lens, captions, caption_lens):
        batch_size = frames.shape[0]
        #print(keywords.shape,texts.shape)
        video_features = self.FramesEncoder(frames, frame_lens)
        video_features_t = torch.sum(video_features,0)
        
        keywords_encode = self.KeywordsEncoder(keywords,keyword_lens)
        keywords_encode_t = torch.sum(keywords_encode,0) 

        text_encode = self.TextsEncoder(texts,texts_lens)
        text_encode_t = torch.sum(text_encode,0) 

        post_mu , post_logvar = self.get_post_latent(keywords_encode_t, text_encode_t, video_features_t)
        prior_mu , prior_logvar = self.get_prior_latent(keywords_encode_t, video_features_t)
        global_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
        
        group_init_state = torch.cat((keywords_encode_t, global_z), 1)
        stop_loss, group_loss = self.GroupDecoder(keywords_encode, segments, group_count, group_init_state)

        bow_loss, local_KL, word_loss = self.Decoder(keywords_encode, video_features, frame_lens, segments, group_count, captions, caption_lens, group_init_state)


        global_KL = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)

        #loss = stop_loss + word_loss + group_loss + global_KL + local_KL + bow_loss


        return stop_loss , word_loss , group_loss , global_KL , local_KL , bow_loss


    def get_prior_latent(self, keywords_encode_t, video_features_t):

        prior_inp = torch.cat((keywords_encode_t, video_features_t), 1)
        prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.config.PHVM_plan_latent_dim,1)
        return prior_mu , prior_logvar

    def get_post_latent(self, keywords_encode_t, text_encode_t, video_features_t):

        post_inp = torch.cat((keywords_encode_t, video_features_t, text_encode_t), 1)
        post_mu , post_logvar = torch.split(self.post_linear2(F.tanh(self.post_linear1(post_inp))),self.config.PHVM_plan_latent_dim,1)
        return post_mu , post_logvar




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
        #segments , group_count= self.GroupDecoder.test(keywords_encode, group_init_state)
        #segments, group_count = self.GroupDecoder.sample(batch_size, keywords_encode, group_init_state)
        
        plans, output_ids = self.Decoder.beam_search(keywords, keywords_encode, video_features, frame_lens, segments_gt, group_count_gt,  group_init_state, self.config.PHVM_beam_width)
        #output_ids = self.Decoder.beam_search(keywords_encode, video_features, frame_lens, segments, group_count,  group_init_state, self.config.PHVM_beam_width)
        

        return plans, output_ids


# Based on tutorials/08 - Language Model
# RNN Based Language Model
class FramesEncoder(nn.Module):
    def __init__(self, config):
        super(FramesEncoder, self).__init__()

        self.config = config
        self.embed_size = config.PHVM_word_dim
        self.vid_dim = config.video_dim
        self.model_dim = config.model_dim
        self.layer = config.video_layer
        self.heads = config.video_heads
        self.enable_cuda = config.cuda

        self.linear = nn.Linear(self.vid_dim, self.model_dim, bias=False)
        self.trans_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.heads)
        self.trans = nn.TransformerEncoder(self.trans_layer, num_layers=self.layer)
        
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.08, 0.08)
        #self.linear.bias.data.fill_(0)

    def forward(self, frames, flengths):
        """Handles variable size frames
           frame_embed: video features
           flengths: frame lengths
        """
        frames = self.linear(frames).transpose(0,1)
        outputs = self.trans(frames)
        return outputs

# Based on tutorials/08 - Language Model
# RNN Based Language Model
class WordsEncoder(nn.Module):
    def __init__(self, config):
        super(WordsEncoder, self).__init__()

        self.config = config
        self.embed_size = config.PHVM_word_dim
        self.vocab_size = config.vocab_size
        self.model_dim = config.model_dim
        self.layer = config.encoder_layer
        self.heads = config.encoder_heads
        self.enable_cuda = config.cuda

        self.linear = nn.Linear(self.embed_size, self.model_dim, bias=False)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.trans_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.heads)
        self.trans = nn.TransformerEncoder(self.trans_layer, num_layers=self.layer)
        
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.08, 0.08)
        #self.linear.bias.data.fill_(0


    def forward(self, inputs, flengths):
        """Handles variable size inputs
           frame_embed: video features
           flengths: frame lengths
        """

        embeddings = self.linear(self.embed(inputs)).transpose(0,1)
        outputs = self.trans(embeddings)

        return outputs

class GroupEncoder(nn.Module):
    def __init__(self, config):
        super(GroupEncoder, self).__init__()

        # self.use_abs = use_abs
        self.config = config

        
        self.model_dim = config.model_dim
        self.max_enc_len = config.max_enc_len
        self.max_group_count = config.max_group_count
        self.plan_latent_dim = config.PHVM_plan_latent_dim
        self.layer = config.group_decoder_layer
        self.heads = config.group_decoder_heads

        self.dropout_rate = config.PHVM_dropout

        self.enable_cuda = config.cuda
        
        self.Linear1 = nn.Linear(self.model_dim+self.plan_latent_dim, self.model_dim,bias=False)
        self.Linear2 = nn.Linear(self.model_dim*2, self.model_dim,bias=False)
        self.trans_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.heads)
        self.trans = nn.TransformerEncoder(self.trans_layer, num_layers=self.layer)

        
        
        self.init_weights()

    def init_weights(self):
        self.Linear1.weight.data.uniform_(-0.08, 0.08)
        self.Linear2.weight.data.uniform_(-0.08, 0.08)


    def forward(self, enc_states, segments, group_count, group_init_state):

        batch_size,step_size,_ = segments.shape

        enc_states = enc_states.transpose(0,1)
        segments_input = torch.cat((Variable(torch.LongTensor(np.zeros([batch_size,1,self.max_enc_len]))).cuda(),segments), 1) #key start group
        inps = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,self.model_dim]))).cuda()
        for i in range(step_size):
            segment = segments_input[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inps[:,i,:] = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)


        inps = self.Linear2(torch.cat((inps.transpose(0,1), self.Linear1(group_init_state.unsqueeze(0).expand(step_size,-1,-1))), 2))

        outputs = self.trans(inps)

        return outputs


class GroupDecoder(nn.Module):
    def __init__(self, config):
        super(GroupDecoder, self).__init__()

        # self.use_abs = use_abs
        self.config = config

        
        self.model_dim = config.model_dim
        self.max_enc_len = config.max_enc_len
        self.max_group_count = config.max_group_count
        self.plan_latent_dim = config.PHVM_plan_latent_dim
        self.layer = config.group_decoder_layer
        self.heads = config.group_decoder_heads

        self.dropout_rate = config.PHVM_dropout

        self.enable_cuda = config.cuda
        
        self.Linear1 = nn.Linear(self.model_dim+self.plan_latent_dim, self.model_dim,bias=False)
        self.Linear2 = nn.Linear(self.model_dim*2, self.model_dim,bias=False)
        self.trans_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.heads)
        self.trans = nn.TransformerEncoder(self.trans_layer, num_layers=self.layer)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.PlanLinear1 = nn.Linear(self.model_dim*2, self.model_dim)
        self.PlanLinear2 = nn.Linear(self.model_dim, 1)
        self.StopLinear = nn.Linear(self.model_dim, 1)
        
        
        self.init_weights()

    def init_weights(self):
        self.Linear1.weight.data.uniform_(-0.08, 0.08)
        self.Linear2.weight.data.uniform_(-0.08, 0.08)
        self.PlanLinear1.weight.data.uniform_(-0.08, 0.08)
        self.PlanLinear1.bias.data.fill_(0)
        self.PlanLinear2.weight.data.uniform_(-0.08, 0.08)
        self.PlanLinear2.bias.data.fill_(0)
        self.StopLinear.weight.data.uniform_(-0.08, 0.08)
        self.StopLinear.bias.data.fill_(0)


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
        enc_states = enc_states.transpose(0,1)


        segments_input = torch.cat((Variable(torch.LongTensor(np.zeros([batch_size,1,self.max_enc_len]))).cuda(),segments), 1) #key start group
        inps = Variable(torch.FloatTensor(np.zeros([batch_size,step_size,self.model_dim]))).cuda()
        for i in range(step_size):
            segment = segments_input[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            inps[:,i,:] = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)


        inps = self.Linear2(torch.cat((inps.transpose(0,1), self.Linear1(group_init_state.unsqueeze(0).expand(step_size,-1,-1))), 2))

        outputs = self.trans(inps).transpose(0,1)#转回batch_first

        if batch_size > 1:
            # Sort by length (keep idx)
            group_count, idx_sort = np.sort(group_count)[::-1], np.argsort(-group_count)
        outputs = outputs.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
        outputs = nn.utils.rnn.pack_padded_sequence(outputs, group_count.copy(), batch_first=True)[0]
        group_targets_packed = nn.utils.rnn.pack_padded_sequence(segments, group_count.copy(), batch_first=True)[0]
    
         #key data_len * max_enc_len
        enc_state_2 = nn.utils.rnn.pack_padded_sequence(enc_states.unsqueeze(1).expand(-1,step_size,-1,-1), group_count.copy(), batch_first=True)[0] #key packed_data_len * max_enc_len * enc_dim
        enc_state_2 = torch.cat((enc_state_2, Variable(torch.FloatTensor(np.zeros([enc_state_2.shape[0],self.max_enc_len-enc_state_2.shape[1],self.model_dim]))).cuda()),1)
        outputs_2 = outputs.unsqueeze(1).expand(-1,self.max_enc_len,-1) #key packed_data_len * max_enc_len * hidden_dim
        outputs_2 = torch.cat((outputs_2, enc_state_2),2)

        stop_label = Variable(torch.zeros(batch_size, step_size)).cuda()
        for i in range(batch_size):
            stop_label[i,group_count[i] -1] = 1

        stop_packed = nn.utils.rnn.pack_padded_sequence(stop_label, group_count.copy(), batch_first=True)[0]

        group_loss = F.binary_cross_entropy_with_logits(self.PlanLinear2(F.tanh(self.PlanLinear1(outputs_2))).squeeze(2), group_targets_packed.float())
        stop_loss = F.binary_cross_entropy_with_logits(self.StopLinear(outputs).squeeze(1), stop_packed.float()) 

        return stop_loss, group_loss

    def sample(self, batch_size, enc_states, group_init_state):

        step_size = self.config.max_group_count

        inp = Variable(torch.FloatTensor(np.zeros([batch_size,1,self.encoder_dim*2]))).cuda()
        state = self.initLinear(group_init_state).unsqueeze(0)
        stop_pred = Variable(torch.FloatTensor(np.zeros([batch_size,step_size]))).cuda()
        segments = Variable(torch.LongTensor(np.zeros([batch_size,step_size,self.max_enc_len]))).cuda()
        group_count = torch.LongTensor(np.zeros(batch_size))
        for i in range(step_size):

            outputs, state = self.rnn(inp, state)
            enc_state_2 = torch.cat((enc_states, Variable(torch.FloatTensor(np.zeros([enc_states.shape[0],self.max_enc_len-enc_states.shape[1],self.encoder_dim*2]))).cuda()),1)
            outputs_2 = outputs.expand(-1,self.max_enc_len,-1) #key batch_size * max_enc_len * hidden_dim
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
        print('group_count:',group_count)

        return segments, group_count  # batch_size*max_step*max_keyword_len, array (batch_size*1)





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        # self.use_abs = use_abs
        self.config = config
        self.model_dim = config.model_dim
        self.sent_decoder_layer = config.sent_decoder_layer
        self.sent_decoder_heads = config.sent_decoder_heads
        self.decoder_layer = config.decoder_layer
        self.decoder_heads = config.decoder_heads
        
        
        self.sent_latent_dim = config.PHVM_sent_latent_dim
        self.video_hid = config.video_hid

        self.bow_hid = config.PHVM_bow_hidden_dim
        self.plan_latent_dim = config.PHVM_plan_latent_dim

        self.embed_size = config.PHVM_word_dim
        self.vocab_size = config.vocab_size

        self.enable_cuda = config.cuda
        self.dropout_rate = config.PHVM_dropout

        self.GroupEncoder = GroupEncoder(config)

        self.SentEncoder = WordsEncoder(config)

        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.sent_trans_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.sent_decoder_heads)
        self.sent_trans = nn.TransformerEncoder(self.sent_trans_layer, num_layers=self.sent_decoder_layer)

        self.trans_layer = nn.TransformerDecoderLayer(d_model=self.model_dim, nhead=self.decoder_heads)
        self.trans = nn.TransformerDecoder(self.trans_layer, num_layers=self.decoder_layer)

        self.bowLinear1 = nn.Linear(config.PHVM_sent_latent_dim+self.model_dim*2, self.bow_hid)
        self.bowLinear2 = nn.Linear(self.bow_hid, self.vocab_size)
        self.wordLinear = nn.Linear(self.model_dim, self.vocab_size)
        self.Linear0 = nn.Linear(self.embed_size, self.model_dim)
        self.Linear1 = nn.Linear(self.model_dim*2+self.sent_latent_dim, self.model_dim)
        self.Linear2 = nn.Linear(self.model_dim*2+self.sent_latent_dim, self.model_dim)
        self.Linear3 = nn.Linear(self.model_dim*2, self.model_dim)

        self.prior_linear1 = nn.Linear(config.model_dim*2, config.PHVM_sent_latent_dim*2)
        self.prior_linear2 = nn.Linear(config.PHVM_sent_latent_dim*2, config.PHVM_sent_latent_dim*2)
        self.post_linear1 = nn.Linear(config.model_dim*3, config.PHVM_sent_latent_dim*2)
        self.post_linear2 = nn.Linear(config.PHVM_sent_latent_dim*2, config.PHVM_sent_latent_dim*2)

        self.ce = nn.CrossEntropyLoss()
        
        self.init_weights()

    def init_weights(self):

        self.bowLinear1.weight.data.uniform_(-0.08, 0.08)
        self.bowLinear1.bias.data.fill_(0)
        self.bowLinear2.weight.data.uniform_(-0.08, 0.08)
        self.bowLinear2.bias.data.fill_(0)
        self.wordLinear.weight.data.uniform_(-0.08, 0.08)
        self.wordLinear.bias.data.fill_(0)
        self.Linear1.weight.data.uniform_(-0.08, 0.08)
        self.Linear1.bias.data.fill_(0)
        self.Linear2.weight.data.uniform_(-0.08, 0.08)
        self.Linear2.bias.data.fill_(0)
        self.Linear3.weight.data.uniform_(-0.08, 0.08)
        self.Linear3.bias.data.fill_(0)

        self.prior_linear1.weight.data.uniform_(-0.08, 0.08)
        self.prior_linear1.bias.data.fill_(0)
        self.prior_linear2.weight.data.uniform_(-0.08, 0.08)
        self.prior_linear2.bias.data.fill_(0)
        self.post_linear1.weight.data.uniform_(-0.08, 0.08)
        self.post_linear1.bias.data.fill_(0)
        self.post_linear2.weight.data.uniform_(-0.08, 0.08)
        self.post_linear2.bias.data.fill_(0)


    def forward(self, enc_states, video_features, frame_lens, segments, group_count, captions, caption_lens, group_init_state):

        batch_size,step_size,max_caption_len = captions.shape

        group_encode = self.GroupEncoder(enc_states, segments, group_count, group_init_state)
        group_encode_t = torch.sum(group_encode,0)
        enc_states = enc_states.transpose(0,1)


        inp = Variable(torch.zeros(batch_size, self.model_dim+self.sent_latent_dim)).cuda()
        inp = self.Linear1(torch.cat((inp, group_encode_t),1)).unsqueeze(0)
        _captions = torch.cat((Variable(torch.LongTensor(np.ones([batch_size,step_size,1]))).cuda(),captions), 2)
        embeddings = self.embed(_captions)

        for i in range(step_size):
            #print('inp.shape',inp.shape)
            sent_out = self.sent_trans(inp).squeeze(0)

            segment = segments[:,i,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            gbow = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)

            sent_encode = self.SentEncoder(captions[:,i,:],caption_lens[:,i])
            sent_encode_t = torch.sum(sent_encode,0)
            post_mu , post_logvar = self.get_post_latent(sent_out, sent_encode_t, gbow)
            prior_mu , prior_logvar = self.get_prior_latent(sent_out, gbow)
            sent_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
            
            output = []
            for k in range(max_caption_len):
                init_states = self.Linear2(torch.cat((sent_z,sent_out,gbow),1))
                word_inp = self.Linear3(torch.cat((self.Linear0(embeddings[:,i,k,:]), init_states), 1)).unsqueeze(1).transpose(0,1)
                word_out = self.trans(word_inp,video_features).squeeze(0)

                output.append(self.wordLinear(word_out))

            inp = self.Linear1(torch.cat((sent_z,word_out,gbow),1)).unsqueeze(0)
            
            output = torch.transpose(torch.stack(output), 0, 1) # converting from step_size x batch_size x vocab_size to batch_size x step_size x vocab_size
            target = captions[:,i,:]

            bow_logit = self.bowLinear2(F.tanh(self.bowLinear1(torch.cat((sent_z, sent_out, gbow),1)))).unsqueeze(1).expand(-1,max_caption_len,-1)
            if batch_size > 1:
            # Sort by length (keep idx)
                caption_len = caption_lens[:,i]
                caption_len, idx_sort = np.sort(caption_len)[::-1], np.argsort(-caption_len)
                if 1 in caption_len:
                    num_data = list(caption_len).index(1)
                else:
                    num_data = len(caption_len)
                if num_data == 0:
                    continue

                output = output.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
                bow_logit = bow_logit.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
                target = target.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
                prior_mu =  prior_mu.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
                prior_logvar = prior_logvar.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
                post_mu = post_mu.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))
                post_logvar = post_logvar.index_select(0, Variable(torch.cuda.LongTensor(idx_sort)))

                caption_len = caption_len[:num_data]
                idx_sort = idx_sort[:num_data]
                output = output[:num_data]
                bow_logit = bow_logit[:num_data]
                target = target[:num_data]

            #print(output.shape,bow_logit.shape,target.shape)
            logit_packed = nn.utils.rnn.pack_padded_sequence(output, caption_len.copy(), batch_first=True)[0]
            bow_logit_packed = nn.utils.rnn.pack_padded_sequence(bow_logit, caption_len.copy(), batch_first=True)[0]
            target_packed = nn.utils.rnn.pack_padded_sequence(target, caption_len.copy(), batch_first=True)[0]
            loss_mask = 1 - torch.from_numpy(caption_lens[:,i].copy()).eq(1).float()
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
        prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.config.PHVM_plan_latent_dim,1)
        return prior_mu , prior_logvar

    def get_post_latent(self, keywords_encode_t, text_encode_t, video_features_t):

        post_inp = torch.cat((keywords_encode_t, video_features_t, text_encode_t), 1)
        post_mu , post_logvar = torch.split(self.post_linear2(F.tanh(self.post_linear1(post_inp))),self.config.PHVM_plan_latent_dim,1)
        return post_mu , post_logvar

    def beam_search(self, keywords, enc_states, video_features, frame_lens, segments, group_count,  group_init_state, beam_size):
        batch_size,step_size,_ = segments.shape
        max_len = self.config.max_caption_len
        group_encode = self.GroupEncoder(enc_states, segments, group_count)

        group_encode_t = Variable(torch.zeros(group_encode.shape[0],group_encode.shape[2])).cuda()
        for i in range(batch_size):
            group_encode_t[i] = group_encode[i, group_count[i]-1,:]  # get last state
        
        sent_state = self.initLinear(torch.cat((group_init_state, group_encode_t), 1)).unsqueeze(0)
        inp = Variable(torch.zeros(batch_size,self.decoder_dim+self.sent_latent_dim)).cuda().unsqueeze(1)
        context_mask = rnn_mask(frame_lens, video_features.shape[1])

        output_ids = []
        #output_ids = []

        for ii in range(step_size):

            sent_hidden,sent_state = self.sentRnn(inp,sent_state)
            #print('sent_hidden:',sent_hidden, 'state:',state)
            hidden_output = sent_hidden.squeeze(1)
            segment = segments[:,ii,:]
            alpha = torch.div(segment.float(), torch.add(segment.sum(1).unsqueeze(1).expand_as(segment),1).float()) #key 归一化
            gbow = torch.bmm(alpha[:,:enc_states.shape[1]].unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)

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
            inp = torch.cat((sent_z,next_hidden),1).unsqueeze(1)
        output_ids = [list(row) for row in zip(*output_ids)]  # 转置
        plans, output_ids = self._agg_group(group_count, output_ids, segments, keywords)
        return plans, output_ids

    def _agg_group(self, stop, text, segments, keywords):
        translation = []
        plans = []
        for gcnt,segs,keys in zip(stop,segments,keywords):
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

        return plans, translation


    



class Attention(nn.Module):
    def __init__(self, config, enc_dim, dec_dim, attn_dim=None):
        super(Attention, self).__init__()
        
        self.config = config
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = self.dec_dim if attn_dim is None else attn_dim


        self.encoder_in = nn.Linear(self.enc_dim, self.attn_dim, bias=True)
        self.decoder_in = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.attn_linear = nn.Linear(self.attn_dim, 1, bias=False)
        self.init_weights()
 

    def init_weights(self):
        self.encoder_in.weight.data.uniform_(-0.08, 0.08)
        self.encoder_in.bias.data.fill_(0)
        self.decoder_in.weight.data.uniform_(-0.08, 0.08)
        self.attn_linear.weight.data.uniform_(-0.08, 0.08)


    def forward(self, dec_state, enc_states, mask, dag=None):
        """
        :param dec_state: 
            decoder hidden state of size batch_size x dec_dim
        :param enc_states:
            all encoder hidden states of size batch_size x max_enc_steps x enc_dim
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

        context_vector = torch.bmm(alpha.unsqueeze(1), enc_states).squeeze(1) # (batch_size, enc_dim)

        return context_vector, alpha


