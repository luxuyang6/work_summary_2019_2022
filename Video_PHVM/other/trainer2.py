from __future__ import print_function

import math
import sys
from glob import glob
import json
import copy
import csv
import torch as t
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable

import data
from utils import *
from models import *
from tensorboard import TensorBoard
from data.common_loader import *

#sys.path.append('../video_caption_eval_python3-master')
from automatic_evaluation import evaluate


logger = get_logger()


def to_var(args, x, volatile=False):
    if args.cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_list_parameters(model):
    if isinstance(model, list):
        params = []
        for m in model:
            params.extend(m.parameters())
    else:
        params = model.parameters()

    return params


def get_optimizer(name):
    if name.lower() == "sgd":
        optim = t.optim.SGD
    elif name.lower() == "adam":
        optim = t.optim.Adam
    elif name.lower() == 'rmsprop':
        optim = t.optim.RMSprop

    return optim




class Trainer(object):
    def __init__(self, args, dataset):
        self.args = args
        self.cuda = args.cuda
        self.dataset = dataset
        self.train_data = dataset['train']
        batcher = self.train_data.get_batcher()
        batch = next(batcher)
        print(batch)
        self.valid_data = dataset['val']
        self.test_data = dataset['test']

        if args.use_tensorboard and args.mode == 'train':
            self.tb = TensorBoard(args.model_dir)  #key?
        else:
            self.tb = None
        self.build_model()

        if self.args.load_path:
            self.load_model()

    def build_model(self):
        self.start_epoch = self.epoch = 0
        self.step = 0
        if self.args.network_type == 'video+plan':
            self.model = PHVM(self.args)
        elif self.args.network_type == 'plan_only':
            self.model = PHVM(self.args)
        else:
            raise NotImplemented(f"Network type `{self.args.network_type}` is not defined")

        self.model.cuda()


        self.ce = nn.CrossEntropyLoss()
        logger.info(f"[*] # Parameters: {self.count_parameters}")

    def build_load_entailment_model(self):
        logger.info(f"Building Entailment model...")
        vocab = data.common_loader.Vocab(self.args.snli_vocab_file, self.args.max_snli_vocab_size)
        self.entailment_data = data.common_loader.SNLIBatcher(self.args.decoder_rnn_max_length, vocab)
        self.entailment_model = CoattMaxPool(self.args)
        if self.args.num_gpu == 1:
            self.entailment_model.cuda()
        self.entailment_model.load_state_dict(
        t.load(self.args.load_entailment_path, map_location=None))
        logger.info(f"[*] LOADED: {self.args.load_entailment_path}")
        


    def train(self):
        optimizer = get_optimizer(self.args.optim)
        self.optim = optimizer(
                self.model.parameters(),
                lr=self.args.lr)

        for self.epoch in range(self.start_epoch, self.args.max_epoch):  
            self.train_model()
            if self.epoch % self.args.save_epoch == 0:
                scores = self.test(mode='val')
                self.save_model(save_criteria_score=scores)


    def train_model(self):
        total_loss = 0
        total_word_loss = 0
        total_WO_KL_loss = 0
        model = self.model
        model.train()
 
        #pbar = tqdm(total=self.train_data.num_steps, desc="train_model")

        batcher = self.train_data.get_batcher()

        for step in range(0,self.train_data.num_steps): 
            batch = next(batcher)

            frames = batch.get('video_batch')
            frame_lens = batch.get('video_len_batch')
            keywords = batch.get('keyword_batch')
            keyword_lens = batch.get('keyword_len_batch')
            texts = batch.get('text_batch')
            texts_lens = batch.get('text_len_batch')
            segments = batch.get('segment_batch')
            captions = batch.get('caption_batch')
            caption_lens = batch.get('caption_len_batch')
            group_count = batch.get('group_count_batch')
            group_lens = batch.get('group_len_batch')

            texts = to_var(self.args, texts)
            segments = to_var(self.args, segments)
            keywords = to_var(self.args, keywords)
            frames = to_var(self.args, frames)
            captions = to_var(self.args, captions)
            stop_loss , word_loss , group_loss , global_KL , local_KL , bow_loss = self.model(frames, frame_lens, keywords, keyword_lens, texts, texts_lens, segments, group_count, group_lens, captions, caption_lens)
            #loss = stop_loss + word_loss + group_loss + global_KL + local_KL + bow_loss
            loss = word_loss
            WO_KL_loss = stop_loss + word_loss + group_loss + bow_loss
            
            self.optim.zero_grad()
            loss.backward()

            t.nn.utils.clip_grad_norm(
                    model.parameters(), self.args.grad_clip)
            self.optim.step()

            total_loss += loss.data
            total_WO_KL_loss += WO_KL_loss.data
            total_word_loss += word_loss
            print(f"train_model| loss: {loss.data.item():5.3f} | step:{step}/{self.train_data.num_steps}")
            #pbar.set_description(f"train_model| loss: {loss.data.item():5.3f}")
            if step % self.args.log_step == 0 and step > 0:
                cur_loss = total_loss.item() / self.args.log_step
                cur_word_loss = total_word_loss.item() / self.args.log_step
                cur_WO_KL_loss = total_WO_KL_loss.item() / self.args.log_step
                ppl = math.exp(cur_word_loss)

                logger.info(f'| epoch {self.epoch:3d} | lr {self.args.lr:8.6f} '
                            f'| loss {cur_loss:.2f} | ppl {ppl:8.2f}')
                            #f'| word loss {cur_word_loss:.2f} | WO_KL_loss {cur_WO_KL_loss:2f}')

                # Tensorboard
                if self.tb is not None:
                    self.tb.scalar_summary("model/loss", cur_loss, self.step)
                    self.tb.scalar_summary("model/perplexity", ppl, self.step)

                total_loss = 0

            step += 1
            self.step += 1

            #pbar.update(1)



    def test(self, mode):

        self.model.eval()
        counter = 0
        if mode == 'val':
            batcher = self.valid_data.get_batcher()
            num_steps = self.valid_data.num_steps
        elif mode == 'test':
            batcher = self.test_data.get_batcher()
            num_steps = self.test_data.num_steps
        else:
            raise Exception("Unknow mode: {}".format(mode))
        
        if mode == 'test':
            f_csv = open(os.path.join(self.args.model_dir,'PredictedCaptions.csv'), 'w', newline='') #output file
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(["Video_ID","Reference","Groud Truth"])


        gts = {}
        res = {}

        for i in range(num_steps):
            batch = next(batcher)
            frames = batch.get('video_batch')
            frame_lens = batch.get('video_len_batch')
            keywords = batch.get('keyword_batch')
            keyword_lens = batch.get('keyword_len_batch')
            segments = batch.get('segment_batch')   
            group_count = batch.get('group_count_batch')
            group_lens = batch.get('group_len_batch')
            segments = to_var(self.args, segments)
            keywords = to_var(self.args, keywords)
            frames = to_var(self.args, frames)

            predicted_targets = self.model.beam_search(frames, frame_lens, keywords, keyword_lens)
            #print(predicted_targets[0])
            #print(predicted_targets[0,0])
            for k,vid in enumerate(batch.get('video_id')):
                caption = [self.test_data.vocab.id2word(id_) for id_ in predicted_targets[k]]
                caption = ' '.join(caption)
                if not caption:
                    caption = '[UNK]'

                res[counter] = [caption]
                gts[counter] = batch.get('text_orgins')[vid]
                
                if mode == 'test':
                    csv_writer.writerow((vid,caption,' '.join(gts[counter])))
                
                counter += 1

        if mode == 'test':
            f_csv.close()
                    

        json.dump(gts,open('gts.json', 'w'))
        json.dump(res,open('res.json', 'w'))
        scores = evaluate(gts, res, score_type='macro', tokenized=True)
        scores_dict = {}
        save_criteria_score = None
        print("Results:")
        for method, score in scores:
            if mode == 'val':
                self.tb.scalar_summary(f"test/{mode}_{method}", score, self.epoch)
            scores_dict[method] = score
            print("{}:{}".format(method,score))
            if self.args.save_criteria == method:
                save_criteria_score = score

        if mode == 'test':
            # save the result
            #if not self.args.load_path.endswith('.pth'):
            if not os.path.exists(os.path.join(self.args.model_dir,'results')):
                os.mkdir(os.path.join(self.args.model_dir,'results'))

            result_save_path = self.result_path
            final_dict = {}
            final_dict['args'] = self.args.__dict__
            final_dict['scores'] = scores_dict
            with open(result_save_path, 'w') as fp:
                json.dump(final_dict, fp, indent=4, sort_keys=True)

        return save_criteria_score


    def calculate_reward(self,sampled_sequence, gts, video_ids, vocab):
        """
        :param sampled_sequence:
            sampled sequence in the form of token_ids of size : batch_size x max_steps
        :param ref_sequence:
            dictionary of reference captions for the given videos
        :param video_ids:
            list of the video_ids
        :param vocab:
            vocab class object used to convert token ids to words
        :param reward_type:
            specify the reward
        :return rewards:
            rewards obtained from the sampled seq w.r.t. ref_seq (metric scores)
        :return seq_lens
            sampled sequence lengths array of size batch_size
        """

        res = {}
        gts_tmp = {}
        seq_lens = []
        batch_size, step_size = sampled_sequence.shape
        counter = 0
        for k in range(batch_size):
            caption = [vocab.id2word(id_) for id_ in sampled_sequence[k,:]]
            # print caption
            punctuation = np.argmax(np.array(caption) == STOP_DECODING)
            if punctuation == 0 and not caption:
                caption = caption
            else: 
                caption = caption[:punctuation]
                caption = ' '.join(caption)

            if not caption:
                caption = UNKNOWN_TOKEN

            res[counter] = [caption]
            gts_tmp[counter] = gts[video_ids[k]]
            counter +=1 
            seq_lens.append(len(caption.split())+1)

        _,reward = evaluate(gts_tmp,res,metric='CIDEr' if self.args.reward_type=='CIDEnt' else self.args.reward_type ,score_type='micro',tokenized=True)[0]
        
        if self.args.reward_type == 'CIDEnt':
            entailment_scores  = self.compute_entailment_scores(gts_tmp, res)


            reward = [x-self.args.lambda_threshold if y<self.args.beta_threshold else x for x,y in zip(reward, entailment_scores)]

        reward = np.array(reward)

        reward = np.reshape(reward,[batch_size,1])

        return reward, np.array(seq_lens)


    def compute_entailment_scores(self,gts,res,length_norm=False):
        scores = []
        for key, value in res.items():
            tmp_prem = gts[key]
            tmp_hypo = [value[0] for _ in range(len(tmp_prem))] 
            batch = self.entailment_data.process_external_data(tmp_prem,tmp_hypo)
            premise = batch.get('premise_batch')
            premise_len = batch.get('premise_length')
            premise = to_var(self.args, premise)
            hypothesis = batch.get('hypothesis_batch')
            hypothesis_len = batch.get('hypothesis_length')
            hypothesis = to_var(self.args, hypothesis)
            self.entailment_model.eval()
            logits, batch_prob, preds = self.entailment_model(premise, premise_len, hypothesis, hypothesis_len)

            batch_prob = batch_prob.cpu().data.numpy()

            scores.append(batch_prob.max())



        return scores


    
    def save_model(self, save_criteria_score=None):
        t.save(self.model.state_dict(), self.path)
        logger.info(f"[*] SAVED: {self.path}")
        epochs, steps  = self.get_saved_models_info()
        
        if save_criteria_score is not None:
            if os.path.exists(os.path.join(self.args.model_dir,'checkpoint_tracker.dat')):
                checkpoint_tracker = t.load(os.path.join(self.args.model_dir,'checkpoint_tracker.dat'))

            else:
                checkpoint_tracker = {}
            key = f"{self.epoch}_{self.step}"
            value = save_criteria_score
            checkpoint_tracker[key] = value
            if len(epochs)>=self.args.max_save_num:
                low_value = 100000.0
                remove_key = None
                for key,value in checkpoint_tracker.items():
                    if low_value > value:
                        remove_key = key
                        low_value = value

                del checkpoint_tracker[remove_key]

                remove_epoch = remove_key.split("_")[0]
                paths = glob(os.path.join(self.args.model_dir,f'*_epoch{remove_epoch}_*.pth'))
                for path in paths:
                    remove_file(path)

            # save back the checkpointer tracker
            t.save(checkpoint_tracker, os.path.join(self.args.model_dir,'checkpoint_tracker.dat'))

        else:
 

            for epoch in epochs[:-self.args.max_save_num]:
                paths = glob(os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pth'))

                for path in paths:
                    remove_file(path)


    def get_saved_models_info(self):
        paths = glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                    name.split(delimiter)[idx].replace(replace_word, ''))
                    for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        steps = get_numbers(basenames, '_', 2, 'step', 'model')

        epochs.sort()
        steps.sort()


        return epochs, steps
    



    def load_model(self):
        
        if self.args.load_path.endswith('.pth'):
            map_location=None
            self.model.load_state_dict(
                t.load(self.args.load_path, map_location=map_location))
            logger.info(f"[*] LOADED: {self.args.load_path}")
        else:
            if os.path.exists(os.path.join(self.args.load_path,'checkpoint_tracker.dat')):
                checkpoint_tracker = t.load(os.path.join(self.args.load_path,'checkpoint_tracker.dat'))
                best_key = None
                best_score = -1.0
                for key,value in checkpoint_tracker.items():
                    if value>best_score:
                        best_score = value
                        best_key = key


                self.epoch = int(best_key.split("_")[0])
                self.step = int(best_key.split("_")[1])

            else:
                epochs, steps = self.get_saved_models_info()

                if len(epochs) == 0:
                    logger.info(f"[!] No checkpoint found in {self.args.model_dir}...")
                    return

                self.epoch = self.start_epoch = max(epochs)
                self.step = max(steps)
            
            if self.args.num_gpu == 0:
                map_location = lambda storage, loc: storage
            else:
                map_location = None

            self.model.load_state_dict(
                    t.load(self.load_path, map_location=map_location))
            logger.info(f"[*] LOADED: {self.load_path}")


    def create_result_path(self, filename):
        return f'{self.args.model_dir}/results/model_epoch{self.epoch}_step{self.step}_{filename}'


    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def path(self):
        return f'{self.args.model_dir}/model_epoch{self.epoch}_step{self.step}.pth'

    @property
    def load_path(self):
        return f'{self.args.load_path}/model_epoch{self.epoch}_step{self.step}.pth'

    @property
    def result_path(self):
        return f'{self.args.model_dir}/results/model_epoch{self.epoch}_step{self.step}.json'

    @property
    def lr(self):
        degree = max(self.epoch - self.args.decay_after + 1, 0)
        return self.args.lr * (self.args.decay ** degree)





