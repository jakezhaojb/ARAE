#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict

class Indexer:
    def __init__(self, symbols = ["<pad>","<unk>","<s>","</s>"]):
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3}
        self.idx2word = {}
        
    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d)

    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.items()]
        items.sort()
        for v, k in items:
            out.write(" ".join([k, str(v)]) + "\n")
        out.close()

    def prune_vocab(self, k, cnt = False):
        vocab_list = [(word, count) for word, count in self.vocab.items()]
        if cnt:
            self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list if pair[1] > k}
        else:
            vocab_list.sort(key = lambda x: x[1], reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d)
        for word, idx in self.d.items():
            self.idx2word[idx] = word

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.strip().split()
            self.d[v] = int(k)
        for word, idx in self.d.items():
            self.idx2word[idx] = word

def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))

def get_data(args):
    indexer = Indexer(["<pad>","<unk>","<s>","</s>"])

    def make_vocab(textfile, seqlength, train=1):
        num_sents = 0
        for sent in open(textfile, 'r'):
            sent = sent.strip().split()
            if len(sent) > seqlength or len(sent) < 1:
                continue
            num_sents += 1
            if train == 1:
                for word in sent:
                    indexer.vocab[word] += 1
        return num_sents

    def convert(textfile, batchsize, seqlength, outfile, num_sents, max_sent_l=0,shuffle=0):
        newseqlength = seqlength + 2 #add 2 for EOS and BOS
        sents = np.zeros((num_sents, newseqlength), dtype=int)
        sent_lengths = np.zeros((num_sents,), dtype=int)
        dropped = 0
        sent_id = 0
        for sent in open(textfile, 'r'):
            sent = [indexer.BOS] + sent.strip().split() + [indexer.EOS]
            max_sent_l = max(len(sent), max_sent_l)
            if len(sent) > seqlength + 2 or len(sent) < 3:
                dropped += 1
                continue
            sent_pad = pad(sent, newseqlength, indexer.PAD)
            sents[sent_id] = np.array(indexer.convert_sequence(sent_pad), dtype=int)
            sent_lengths[sent_id] = (sents[sent_id] != 0).sum()
            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))
        print(sent_id, num_sents)
        if shuffle == 1:
            rand_idx = np.random.permutation(sent_id)
            sents = sents[rand_idx]
            sent_lengths = sent_lengths[rand_idx]

        #break up batches based on source lengths
        sent_lengths = sent_lengths[:sent_id]
        sent_sort = np.argsort(sent_lengths)
        sents = sents[sent_sort]
        sent_l = sent_lengths[sent_sort]
        curr_l = 1
        l_location = [] #idx where sent length changes

        for j,i in enumerate(sent_sort):
            if sent_lengths[i] > curr_l:
                curr_l = sent_lengths[i]
                l_location.append(j)
        l_location.append(len(sents))
        #get batch sizes
        curr_idx = 0
        batch_idx = [0]
        nonzeros = []
        batch_l = []
        batch_w = []
        for i in range(len(l_location)-1):
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1):
            batch_l.append(batch_idx[i+1] - batch_idx[i])
            batch_w.append(sent_l[batch_idx[i]])

        # Write output
        f = h5py.File(outfile, "w")

        f["source"] = sents
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["source_l"] = np.array(batch_w, dtype=int)
        f["sents_l"]  = np.array(sent_l, dtype = int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["vocab_size"] = np.array([len(indexer.d)])
        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()
        return max_sent_l

    print("First pass through data to get vocab...")
    num_sents_train = make_vocab(args.trainfile, args.seqlength)
    print("Number of sentences in training: {}".format(num_sents_train))
    num_sents_valid = make_vocab(args.valfile, args.seqlength, 0)
    print("Number of sentences in valid: {}".format(num_sents_valid))
    num_sents_test = make_vocab(args.testfile, args.seqlength, 0)
    print("Number of sentences in test: {}".format(num_sents_test))    
    if args.vocabminfreq >= 0:
        indexer.prune_vocab(args.vocabminfreq, True)        
    else:
        indexer.prune_vocab(args.vocabsize, False)
    if args.vocabfile != '':
        print('Loading pre-specified source vocab from ' + args.vocabfile)
        indexer.load_vocab(args.vocabfile)
    indexer.write(args.outputfile + ".dict")
    print("Vocab size: Original = {}, Pruned = {}".format(len(indexer.vocab),
                                                          len(indexer.d)))
    max_sent_l = 0
    max_sent_l = convert(args.valfile, args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_sents_valid,
                         max_sent_l, args.shuffle)
    max_sent_l = convert(args.testfile, args.batchsize, args.seqlength,
                         args.outputfile + "-test.hdf5", num_sents_test,
                         max_sent_l, args.shuffle)
    max_sent_l = convert(args.trainfile, args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_sents_train,
                         max_sent_l, args.shuffle)
    print("Max sent length (before dropping): {}".format(max_sent_l))

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                " Rest are replaced with special UNK tokens.",
                                                type=int, default=70000)
    parser.add_argument('--vocabminfreq', help="Minimum frequency for vocab, if using frequency cutoff",
                                                type=int, default=-1)    
    parser.add_argument('--trainfile', help="Path to training data.", required=True)
    parser.add_argument('--valfile', help="Path validation data.", required=True)
    parser.add_argument('--testfile', help="Path to test data.", required=True)    
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=32)
    parser.add_argument('--seqlength', help="Maximum source sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=200)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str)
    parser.add_argument('--vocabfile', help="If working with a preset vocab, "
                                          "then including this will ignore srcvocabsize and use the"
                                          "vocab provided here.",
                                          type = str, default='')
    parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on  "
                                           "source length).",
                                          type = int, default = 1)

    args = parser.parse_args(arguments)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
