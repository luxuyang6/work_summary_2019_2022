#!/usr/bin/env python

import torch
torch.cuda.init()
torch.cuda.current_device()
torch.cuda._initialized = True
import data
import models
import config
from utils import *
from trainer import  Trainer
from utils import get_logger

logger = get_logger()


def main(args):
    prepare_dirs(args)
    print(args.validate)
    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

#加载数据集

    vocab = data.common_loader.Vocab(args.vocab_file, args.max_vocab_size)
    dataset = {}

    dataset['train'] = data.common_loader.ChaBatcher(args, 'train', vocab)
    dataset['val'] = data.common_loader.ChaBatcher(args, 'val', vocab)
    dataset['test'] = data.common_loader.ChaBatcher(args, 'test', vocab)

#训练器trainer
    trainer = Trainer(args, dataset)
    trainer.test_fold(args.mode)

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)
