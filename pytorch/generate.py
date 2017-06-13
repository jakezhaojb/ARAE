import argparse
import os
import numpy as np
import random
import json

import torch
from torch.autograd import Variable

from models import Seq2Seq, MLP_D, MLP_G

###############################################################################
# Generation methods
###############################################################################


def generate(z, vocab):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        noise = Variable(z, volatile=True)
    elif type(z) == np.ndarray:
        noise = Variable(torch.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=model_args['maxlen'],
                                       sample=args.sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences


def interpolate(z1, z2, vocab, steps=5):
    """
    Interpolating in z space
    Assumes that type(z1) == type(z2)
    """
    if type(z1) == Variable:
        noise1 = z1
        noise2 = z2
    elif type(z1) == torch.FloatTensor or type(z1) == torch.cuda.FloatTensor:
        noise1 = Variable(z1, volatile=True)
        noise2 = Variable(z2, volatile=True)
    elif type(z1) == np.ndarray:
        noise1 = Variable(torch.from_numpy(z1).float(), volatile=True)
        noise2 = Variable(torch.from_numpy(z2).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z1)))

    # interpolation weights
    lambdas = [x*1.0/(steps-1) for x in range(steps)]

    gens = []
    for L in lambdas:
        gens.append(generate((1-L)*noise1 + L*noise2, vocab))

    interpolations = []
    for i in range(len(gens[0])):
        interpolations.append([s[i] for s in gens])

    return interpolations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    parser.add_argument('--load_path', type=str, required=True,
                        help='directory to load models from')
    parser.add_argument('--temp', type=float, default=1,
                        help='softmax temperature (lower --> more discrete)')
    parser.add_argument('--ngenerations', type=int, default=10,
                        help='Number of sentences to generate')
    parser.add_argument('--ninterpolations', type=int, default=5,
                        help='Number z-space sentence interpolation examples')
    parser.add_argument('--steps', type=int, default=5,
                        help='Number of steps in each interpolation')
    parser.add_argument('--outf', type=str, default='./generated.txt',
                        help='filename and path to write to')
    parser.add_argument('--noprint', action='store_true',
                        help='prevents examples from printing')
    parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    print(vars(args))

    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    ###########################################################################
    # Load the models
    ###########################################################################

    model_args = json.load(open("{}/args.json".format(args.load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(args.load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    autoencoder = Seq2Seq(emsize=model_args['emsize'],
                          nhidden=model_args['nhidden'],
                          ntokens=model_args['ntokens'],
                          nlayers=model_args['nlayers'],
                          hidden_init=model_args['hidden_init'])
    gan_gen = MLP_G(ninput=model_args['z_size'],
                    noutput=model_args['nhidden'],
                    layers=model_args['arch_g'])
    gan_disc = MLP_D(ninput=model_args['nhidden'],
                     noutput=1,
                     layers=model_args['arch_d'])

    print('Loading models from'+args.load_path)
    ae_path = os.path.join(args.load_path, "autoencoder_model.pt")
    gen_path = os.path.join(args.load_path, "gan_gen_model.pt")
    disc_path = os.path.join(args.load_path, "gan_disc_model.pt")

    autoencoder.load_state_dict(torch.load(ae_path))
    gan_gen.load_state_dict(torch.load(gen_path))
    gan_disc.load_state_dict(torch.load(disc_path))

    ###########################################################################
    # Generation code
    ###########################################################################

    # Generate sentences
    if args.ngenerations > 0:
        noise = torch.ones(args.ngenerations, model_args['z_size'])
        noise.normal_()
        sentences = generate(z=noise, vocab=idx2word)

        if not args.noprint:
            print("\nSentence generations:\n")
            for sent in sentences:
                print(sent)
        with open(args.outf, "w") as f:
            f.write("Sentence generations:\n\n")
            for sent in sentences:
                f.write(sent+"\n")

    # Generate interpolations
    if args.ninterpolations > 0:
        noise1 = torch.ones(args.ninterpolations, model_args['z_size'])
        noise1.normal_()
        noise2 = torch.ones(args.ninterpolations, model_args['z_size'])
        noise2.normal_()
        interps = interpolate(z1=noise1,
                              z2=noise2,
                              vocab=idx2word,
                              steps=args.steps)

        if not args.noprint:
            print("\nSentence interpolations:\n")
            for interp in interps:
                for sent in interp:
                    print(sent)
                print("")
        with open(args.outf, "a") as f:
            f.write("\nSentence interpolations:\n\n")
            for interp in interps:
                for sent in interp:
                    f.write(sent+"\n")
                f.write('\n')
