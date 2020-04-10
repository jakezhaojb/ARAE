#!/usr/bin/env python3

import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import h5py
import time
import logging

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_file', default='')
parser.add_argument('--val_file', default='')
parser.add_argument('--train_from', default='')

# Model options
parser.add_argument('--word_dim', default=300, type=int)
parser.add_argument('--h_dim', default=300, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0.2, type=float)

# Optimization options
parser.add_argument('--checkpoint_path', default='baseline.pt')
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--lr', default=1, type=float)
parser.add_argument('--max_grad_norm', default=5, type=float)
parser.add_argument('--test', default=0, type=int)
parser.add_argument('--gpu', default=2, type=int)
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--print_every', type=int, default=500)

class Dataset(object):
  def __init__(self, h5_file):
    data = h5py.File(h5_file, 'r') 
    self.sents = self._convert(data['source']).long()
    self.sent_lengths = self._convert(data['source_l']).long()
    self.batch_size = self._convert(data['batch_l']).long()
    self.batch_idx = self._convert(data['batch_idx']).long()
    self.vocab_size = data['vocab_size'][0]
    self.num_batches = self.batch_idx.size(0)

  def _convert(self, x):
    return torch.from_numpy(np.asarray(x))

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    assert(idx < self.num_batches and idx >= 0)
    start_idx = self.batch_idx[idx]
    end_idx = start_idx + self.batch_size[idx]
    length = self.sent_lengths[idx]
    sents = self.sents[start_idx:end_idx]
    batch_size = self.batch_size[idx]
    data_batch = [Variable(sents[:, :length]), length-1, batch_size]
    return data_batch

class RNNLM(nn.Module):
  def __init__(self, vocab_size=10000,
               word_dim=300,
               h_dim=300,
               num_layers=1,
               dropout=0):
    super(RNNLM, self).__init__()
    self.h_dim = h_dim
    self.num_layers = num_layers    
    self.word_vecs = nn.Embedding(vocab_size, word_dim)
    self.dropout = nn.Dropout(dropout)
    self.rnn = nn.LSTM(word_dim, h_dim, num_layers = num_layers,
                       dropout = dropout, batch_first = True)      
    self.vocab_linear = nn.Sequential(nn.Dropout(dropout), 
                                      nn.Linear(h_dim, vocab_size),
                                        nn.LogSoftmax(dim=-1))  
  def forward(self, sent):
    word_vecs = self.dropout(self.word_vecs(sent[:, :-1])) #last token is </s>
    h, _ = self.rnn(word_vecs)
    preds = self.vocab_linear(h)
    return preds

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset(args.train_file)
  val_data = Dataset(args.val_file)  
  train_sents = train_data.batch_size.sum()
  vocab_size = int(train_data.vocab_size)    
  
  print('Train data: %d batches' % len(train_data))
  print('Val data: %d batches' % len(val_data))
  print('Word vocab size: %d' % vocab_size)
  cuda.set_device(args.gpu)
    
  if args.train_from == '':
    model = RNNLM(vocab_size = vocab_size,
                  word_dim = args.word_dim,
                  h_dim = args.h_dim,
                  num_layers = args.num_layers,
                  dropout = args.dropout)
    for param in model.parameters():    
      param.data.uniform_(-0.1, 0.1)      
  else:
    print('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']
  print("model architecture")
  print(model)
  
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  criterion = nn.NLLLoss()  
  model.train()

  if args.gpu >= 0:
    model.cuda()
    criterion.cuda()
    
  best_val_ppl = 1e5
  epoch = 0
  if args.test == 1:
    print('Evaluating on test')
    eval(val_data, model, criterion)
    exit()
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    num_sents = 0
    num_words = 0
    b = 0

    for i in np.random.permutation(len(train_data)):
      sents, length, batch_size = train_data[i]
      if args.gpu >= 0:
        sents = sents.cuda()
      b += 1
      optimizer.zero_grad()
      preds = model(sents)
      nll = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      train_nll += nll.data[0]*batch_size
      nll.backward()
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)        
      optimizer.step()

      num_sents += batch_size
      num_words += batch_size * length
      
      if b % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).data[0]**0.5
        print('Epoch: %d, Batch: %d/%d, LR: %.4f, TrainPPL: %.2f, |Param|: %.4f, BestValPerf: %.2f, Throughput: %.2f examples/sec' % 
              (epoch, b, len(train_data), args.lr, np.exp(train_nll / num_words), 
               param_norm, best_val_ppl, num_sents / (time.time() - start_time)))
    print('--------------------------------')
    print('Checking validation perf...')
    val_ppl  = eval(val_data, model, criterion)
    if val_ppl < best_val_ppl:
      best_val_ppl = val_ppl
      checkpoint = {
        'args': args.__dict__,
        'model': model,
        'optimizer': optimizer
      }
      print('Saving checkpoint to %s' % args.checkpoint_path)
      torch.save(checkpoint, args.checkpoint_path)

def eval(data, model, criterion):
  model.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  for i in range(len(data)):
    sents, length, batch_size = data[i]
    num_words += batch_size*length
    num_sents += batch_size
    if args.gpu >= 0:
      sents = sents.cuda()
    preds = model.forward(sents)
    nll = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
    total_nll += nll.data[0]*batch_size
  ppl = np.exp(total_nll / num_words)
  print('PPL: %.4f' % (ppl))
  model.train()
  return ppl

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
