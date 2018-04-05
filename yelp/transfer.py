import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json
import kenlm
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify, truncate, get_ppl
from models import Seq2Seq2Decoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify, load_models

parser = argparse.ArgumentParser(description='PyTorch ARAE for Text')
# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--load_path', type=str, required=True,
                    help='path to load model from')
parser.add_argument('--epoch', type=int, required=True,
                    help='epoch')
parser.add_argument('--load_vocab', type=str, default="",
                    help='path to load vocabulary from')

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=50000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=30,
                    help='maximum sentence length')
parser.add_argument('--lowercase', action='store_true',
                    help='lowercase all text')

# Evaluation Arguments
parser.add_argument('--batch_size', type=int, default=32,
                    help='batc size')
parser.add_argument('--lm_path', type=str, default="", #TODO
                    help='language model path')
parser.add_argument('--ft_path', type=str, default="", #TODO
                    help='language model path')
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--debug', action='store_true',
                    help='debug')
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
print(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if not os.path.isdir('{}/eval'.format(args.load_path)):
    os.makedirs('{}/eval'.format(args.load_path))

###############################################################################
# Load data
###############################################################################

# (Path to textfile, Name, Use4Vocab)
datafiles = [(os.path.join(args.data_path, "test1.txt"), "test1", False),
             (os.path.join(args.data_path, "test2.txt"), "test2", True)]
if args.load_vocab != "":
    vocabdict = json.load(args.vocab)
else:
    vocabdict = json.load(open(os.path.join(args.load_path, "vocab.json"), 'r'))
vocabdict = {k: int(v) for k, v in vocabdict.items()}

corpus = Corpus(datafiles,
                maxlen=args.maxlen,
                vocab_size=len(vocabdict),
                lowercase=args.lowercase,
                vocab=vocabdict,
                debug=args.debug)

eval_batch_size = args.batch_size
test1_data = batchify(corpus.data['test1'], eval_batch_size, shuffle=False)
test2_data = batchify(corpus.data['test2'], eval_batch_size, shuffle=False)
print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

model_args, idx2word, autoencoder, gan_gen, gan_disc = \
        load_models(args.load_path, args.epoch, twodecoders=True)

ntokens = len(corpus.dictionary.word2idx)

if args.cuda:
    autoencoder = autoencoder.cuda()
    autoencoder.gpu = True

###############################################################################
# Training code
###############################################################################

def evaluate_transfer(whichdecoder, data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    ntokens = len(corpus.dictionary.word2idx)
    
    original = []
    transferred = []
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        target = target.view(source.size(0), -1)
        source = to_gpu(args.cuda, Variable(source, volatile=True))

        mask = target.gt(0)
        hidden = autoencoder(0, source, lengths, noise=False, encode_only=True)

        # output: batch x seq_len x ntokens
        if whichdecoder == 1:
            max_indices = autoencoder.generate(2, hidden, maxlen=args.maxlen)
        else:
            max_indices = autoencoder.generate(1, hidden, maxlen=args.maxlen)
            
        for t, idx in zip(target, max_indices):
            t = t.numpy()
            idx = idx.data.cpu().numpy()
            
            words = [corpus.dictionary.idx2word[x] for x in t]
            original.append(truncate(words))

            words = [corpus.dictionary.idx2word[x] for x in idx]
            transferred.append(truncate(words))

    return original, transferred

print("Evaluating transfer")
original1, transfer1 = evaluate_transfer(1, test1_data, args.epoch)
original2, transfer2 = evaluate_transfer(2, test2_data, args.epoch)

print("Writing results")
original_file = "{}/eval/original_epoch{}.txt".format(args.load_path, args.epoch)
with open(original_file, 'w') as f:
    for sent in original1:
        f.write(sent+"\n")
    for sent in original2:
        f.write(sent+"\n")

transfer_file = "{}/eval/transfer_epoch{}.txt".format(args.load_path, args.epoch)
with open(transfer_file, 'w') as f:
    for sent in transfer1:
        f.write(sent+"\n")
    for sent in transfer2:
        f.write(sent+"\n")

ft_file = "{}/eval/sentiment_epoch{}.ft".format(args.load_path, args.epoch)
with open(ft_file, 'w') as f:
    for sent in transfer1:
        f.write("__label__2 "+sent+"\n")
    for sent in transfer2:
        f.write("__label__1 "+sent+"\n")

# Perplexity
model = kenlm.Model(args.lm_path)
ppl = get_ppl(model, transfer1+transfer2)
print("Perplexity: {}".format(ppl))

# BLEU
BLEU_CMD = "/home/kz918/ARAE-dev/yelp/tool/multi-bleu.perl -lc {} < {}".format(original_file, transfer_file)

# FastText
FT_CMD = "cd /home/kz918/fastText; ./fasttext test {} {} 1".format(args.ft_path, ft_file)

print("\nFast Text")
result = subprocess.check_output(FT_CMD, shell=True)
#os.system(FT_CMD)
print(result)

print("\nBLEU")
result = subprocess.check_output(BLEU_CMD, shell=True)
#os.system(BLEU_CMD)
print(result)
