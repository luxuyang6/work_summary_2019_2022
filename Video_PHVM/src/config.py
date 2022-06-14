import argparse
from utils import get_logger

logger = get_logger()

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')




#network
parser.add_argument('--birnn', type=str2bool, default=True)
parser.add_argument('--network_type', type=str, choices=['GRU','GRU_plan','Transformer','GRU_wo_video'], default='GRU')

#video
parser.add_argument('--max_vid_len', type=int, default=150)
parser.add_argument('--vid_dim', type=int, default=2048)
parser.add_argument('--vid_hid', type=int, default=1024)
parser.add_argument('--vid_layer', type=int, default=1)

#embedding
parser.add_argument('--max_vocab_size', type=int, default=23000)
parser.add_argument('--vocab_size', type=int, default=4118)
parser.add_argument('--share_vocab', type=str2bool, default=True)
parser.add_argument('--embed_size', type=int, default=300)

#group
parser.add_argument('--group_selection_threshold', type=float, default=0.3)
parser.add_argument('--stop_threshold', type=float, default=0.6)

#encoder
parser.add_argument('--enc_dim', type=int, default=100)
parser.add_argument('--max_enc_len', type=int, default=35)
parser.add_argument('--enc_layer', type=int, default=1)

#group_decoder
parser.add_argument('--max_group_count', type=int, default=10)
parser.add_argument('--group_decoder_dim', type=int, default=100)
parser.add_argument('--group_decoder_layer', type=int, default=1)

#group_encoder
parser.add_argument('--group_encoder_dim', type=int, default=100)
parser.add_argument('--group_encoder_layer', type=int, default=1)

#sent_decoder
parser.add_argument('--sent_decoder_dim', type=int, default=300)
parser.add_argument('--sent_decoder_layer', type=int, default=1)

#text_encoder
parser.add_argument('--text_encoder_dim', type=int, default=300)
parser.add_argument('--text_encoder_layer', type=int, default=1)

#bow
parser.add_argument('--bow_hid', type=int, default=200)
# parser.add_argument('--initial', type=str2bool, default=True)

#decoder
parser.add_argument('--decoder_dim', type=int, default=300)
parser.add_argument('--decoder_layer', type=int, default=2)
parser.add_argument('--max_text_len', type=int, default=120)
parser.add_argument('--max_dec_len', type=int, default=30)
parser.add_argument('--use_gt_plan', type=str2bool, default=True)

#latent
parser.add_argument('--plan_latent_dim', type=int, default=200)
parser.add_argument('--sent_latent_dim', type=int, default=200)


#Transformer
parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--video_trans_layer', type=int, default=4)
parser.add_argument('--video_heads', type=int, default=8)
parser.add_argument('--encoder_trans_layer', type=int, default=4)
parser.add_argument('--encoder_heads', type=int, default=8)
parser.add_argument('--group_decoder_trans_layer', type=int, default=1)
parser.add_argument('--group_decoder_heads', type=int, default=8)
parser.add_argument('--group_encoder_trans_layer', type=int, default=1)
parser.add_argument('--group_encoder_heads', type=int, default=8)
parser.add_argument('--sent_decoder_trans_layer', type=int, default=1)
parser.add_argument('--sent_decoder_heads', type=int, default=8)
parser.add_argument('--decoder_trans_layer', type=int, default=4)
parser.add_argument('--decoder_heads', type=int, default=8)
parser.add_argument('--position_encoding', type=str2bool, default=True)
parser.add_argument('--max_relative_positions', type=int, default=0)
parser.add_argument('--transformer_ff', type=int, default=2048)
parser.add_argument('--attention_dropout', type=float, default=0.1)




#training & testing
parser.add_argument('--validate', type=str2bool, default=True)
parser.add_argument('--new_test', type=str2bool, default=True)
parser.add_argument('--test_fold', type=str2bool, default=False)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'val'])

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--test_batch_size', type=int, default=8)
parser.add_argument('--loss_function', type=str, default='xe', choices=['xe','rl', 'xe+rl'])
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--grad_clip', type=float, default=10.0)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--use_decay_lr', type=str2bool, default=False)
parser.add_argument('--decay', type=float, default=0.96)
parser.add_argument('--lambda_threshold', type=float, default=0.5)
parser.add_argument('--beta_threshold', type=float, default=0.333)

parser.add_argument('--print_result', type=str2bool, default=False)
parser.add_argument('--random_plan', type=str2bool, default=False)
parser.add_argument('--random_min_sent_len', type=int, default=3)
parser.add_argument('--random_min_word_len', type=int, default=3)
parser.add_argument('--random_max_sent_len', type=int, default=3)
parser.add_argument('--random_max_word_len', type=int, default=6)
parser.add_argument('--shuffle_plan', type=str2bool, default=False)
parser.add_argument('--full_plan', type=str2bool, default=False)
parser.add_argument('--plan_decay', type=float, default=1.0)
parser.add_argument('--plan_dim_decay', type=int, default=100)
parser.add_argument('--plan_decay_dim', type=int, default=100)
parser.add_argument('--change1', type=str2bool, default=False)

parser.add_argument('--anneal', type=str2bool, default=False)
parser.add_argument('--one2many', type=int, default=1)
parser.add_argument('--gumbel', type=str2bool, default=False)
parser.add_argument('--gumbel_temperature', type=float, default=1.0)

# Data

parser.add_argument('--dataset', type=str, default='charades')
parser.add_argument('--vid_feature_path', type=str, default='../data/Charades_feature')
parser.add_argument('--train_file', type=str, default='../data/preprocess/Charades_v1_train.json')
parser.add_argument('--test_file', type=str, default='../data/preprocess/Charades_v1_test.json')
parser.add_argument('--test_plan_file', type=str, default='../logs/charades_PHVM_GRU_only_plan_4_t_3_6/model_epoch190_step357552_all.csv')
#parser.add_argument('--test_plan_file', type=str, default='../data/preprocess/selected_plan_2.json')
parser.add_argument('--test_index', type=int, default=0)
parser.add_argument('--val_file', type=str, default='../data/preprocess/Charades_v1_val.json')
parser.add_argument('--vocab_file', type=str, default='../data/preprocess/wordlist.json')

#other
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--save_criteria', type=str, default='AVG', choices=['CIDEr', 'AVG','BLEU-4','ROUGE_L'])
parser.add_argument('--max_save_num', type=int, default=5)
parser.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
parser.add_argument('--log_dir', type=str, default='../logs')
parser.add_argument('--data_dir', type=str, default='../data')
#parser.add_argument('--model_dir', type=str, default='../result')
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=1111)
parser.add_argument('--use_tensorboard', type=str2bool, default=True)
parser.add_argument('--weight_init', type=float, default=None)
parser.add_argument('--cell_type', type=str, default='lstm', choices=['lstm','gru'])


def get_args():
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed
