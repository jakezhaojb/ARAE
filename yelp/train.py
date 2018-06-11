import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify
from models import Seq2Seq2Decoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify
import shutil

parser = argparse.ArgumentParser(description='ARAE for Yelp transfer')
# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--outf', type=str, default='yelp_example',
                    help='output directory name')
parser.add_argument('--load_vocab', type=str, default="",
                    help='path to load vocabulary from')

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=30000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=25,
                    help='maximum sentence length')
parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                    help='lowercase all text')
parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
                    help='not lowercase all text')
parser.set_defaults(lowercase=True)

# Model Arguments
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_r', type=float, default=0.1,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.9995,
                    help='anneal noise_r exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='128-128',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='128-128',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--arch_classify', type=str, default='128-128',
                    help='classifier architecture')
parser.add_argument('--z_size', type=int, default=32,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')

# Training Arguments
parser.add_argument('--epochs', type=int, default=25,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_gan_ae', type=int, default=1,
                    help='number of gan-into-ae iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=1e-04,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-04,
                    help='critic/discriminator learning rate')
parser.add_argument('--lr_classify', type=float, default=1e-04,
                    help='classifier learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')
parser.add_argument('--gan_gp_lambda', type=float, default=0.1,
                    help='WGAN GP penalty lambda')
parser.add_argument('--grad_lambda', type=float, default=0.01,
                    help='WGAN into AE lambda')
parser.add_argument('--lambda_class', type=float, default=1,
                    help='lambda on classifier')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--no-cuda', dest='cuda', action='store_true',
                    help='not using CUDA')
parser.set_defaults(cuda=True)
parser.add_argument('--debug', action='store_true',
                    help='debug')
parser.add_argument('--device_id', type=str, default='0')

args = parser.parse_args()
print(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

# make output directory if it doesn't already exist
if os.path.isdir(args.outf):
    shutil.rmtree(args.outf)
os.makedirs(args.outf)

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

###############################################################################
# Load data
###############################################################################

label_ids = {"pos": 1, "neg": 0}
id2label = {1:"pos", 0:"neg"}

# (Path to textfile, Name, Use4Vocab)
datafiles = [(os.path.join(args.data_path, "valid1.txt"), "valid1", False),
             (os.path.join(args.data_path, "valid2.txt"), "valid2", False),
             (os.path.join(args.data_path, "train1.txt"), "train1", True),
             (os.path.join(args.data_path, "train2.txt"), "train2", True)]
if args.debug:
    datafiles = datafiles[:2]
vocabdict = None
if args.load_vocab != "":
    vocabdict = json.load(args.vocab)
    vocabdict = {k: int(v) for k, v in vocabdict.items()}
corpus = Corpus(datafiles,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                vocab=vocabdict,
                debug=args.debug)

# dumping vocabulary
with open('{}/vocab.json'.format(args.outf), 'w') as f:
    json.dump(corpus.dictionary.word2idx, f)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens
with open('{}/args.json'.format(args.outf), 'w') as f:
    json.dump(vars(args), f)
with open("{}/log.txt".format(args.outf), 'w') as f:
    f.write(str(vars(args)))
    f.write("\n\n")

eval_batch_size = 100
test1_data = batchify(corpus.data['valid1'], eval_batch_size, shuffle=False)
test2_data = batchify(corpus.data['valid2'], eval_batch_size, shuffle=False)
if args.debug:
    train1_data = batchify(corpus.data['valid1'], args.batch_size, shuffle=True)
    train2_data = batchify(corpus.data['valid2'], args.batch_size, shuffle=True)
else:
    train1_data = batchify(corpus.data['train1'], args.batch_size, shuffle=True)
    train2_data = batchify(corpus.data['train2'], args.batch_size, shuffle=True)

print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

ntokens = len(corpus.dictionary.word2idx)
autoencoder = Seq2Seq2Decoder(emsize=args.emsize,
                      nhidden=args.nhidden,
                      ntokens=ntokens,
                      nlayers=args.nlayers,
                      noise_r=args.noise_r,
                      hidden_init=args.hidden_init,
                      dropout=args.dropout,
                      gpu=args.cuda)

gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
g_factor = None

print(autoencoder)
print(gan_gen)
print(gan_disc)
print(classifier)

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))
#### classify
optimizer_classify = optim.Adam(classifier.parameters(),
                                lr=args.lr_classify,
                                betas=(args.beta1, 0.999))

criterion_ce = nn.CrossEntropyLoss()

if args.cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    classifier = classifier.cuda()
    criterion_ce = criterion_ce.cuda()

###############################################################################
# Training code
###############################################################################


def save_model():
    print("Saving models")
    with open('{}/autoencoder_model.pt'.format(args.outf), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('{}/gan_gen_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('{}/gan_disc_model.pt'.format(args.outf), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)


def train_classifier(whichclass, batch):
    classifier.train()
    classifier.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    labels = to_gpu(args.cuda, Variable(torch.zeros(source.size(0)).fill_(whichclass-1)))

    # Train
    code = autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
    scores = classifier(code)
    classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
    classify_loss.backward()
    optimizer_classify.step()
    classify_loss = classify_loss.cpu().data[0]

    pred = scores.data.round().squeeze(1)
    accuracy = pred.eq(labels.data).float().mean()

    return classify_loss, accuracy


def grad_hook_cla(grad):
    return grad * args.lambda_class


def classifier_regularize(whichclass, batch):
    autoencoder.train()
    autoencoder.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))
    flippedclass = abs(2-whichclass)
    labels = to_gpu(args.cuda, Variable(torch.zeros(source.size(0)).fill_(flippedclass)))

    # Train
    code = autoencoder(0, source, lengths, noise=False, encode_only=True)
    code.register_hook(grad_hook_cla)
    scores = classifier(code)
    classify_reg_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
    classify_reg_loss.backward()

    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    return classify_reg_loss


def evaluate_autoencoder(whichdecoder, data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        hidden = autoencoder(0, source, lengths, noise=False, encode_only=True)

        # output: batch x seq_len x ntokens
        if whichdecoder == 1:
            output = autoencoder(1, source, lengths, noise=False)
            flattened_output = output.view(-1, ntokens)
            masked_output = \
                flattened_output.masked_select(output_mask).view(-1, ntokens)
            # accuracy
            max_vals1, max_indices1 = torch.max(masked_output, 1)
            all_accuracies += \
                torch.mean(max_indices1.eq(masked_target).float()).data[0]
        
            max_values1, max_indices1 = torch.max(output, 2)
            max_indices2 = autoencoder.generate(2, hidden, maxlen=50)
        else:
            output = autoencoder(2, source, lengths, noise=False)
            flattened_output = output.view(-1, ntokens)
            masked_output = \
                flattened_output.masked_select(output_mask).view(-1, ntokens)
            # accuracy
            max_vals2, max_indices2 = torch.max(masked_output, 1)
            all_accuracies += \
                torch.mean(max_indices2.eq(masked_target).float()).data[0]

            max_values2, max_indices2 = torch.max(output, 2)
            max_indices1 = autoencoder.generate(1, hidden, maxlen=50)
        
        total_loss += criterion_ce(masked_output/args.temp, masked_target).data
        bcnt += 1

        aeoutf_from = "{}/{}_output_decoder_{}_from.txt".format(args.outf, epoch, whichdecoder)
        aeoutf_tran = "{}/{}_output_decoder_{}_tran.txt".format(args.outf, epoch, whichdecoder)
        with open(aeoutf_from, 'w') as f_from, open(aeoutf_tran,'w') as f_trans:
            max_indices1 = \
                max_indices1.view(output.size(0), -1).data.cpu().numpy()
            max_indices2 = \
                max_indices2.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            tran_indices = max_indices2 if whichdecoder == 1 else max_indices1
            for t, tran_idx in zip(target, tran_indices):
                # real sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f_from.write(chars)
                f_from.write("\n")
                # transfer sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in tran_idx])
                f_trans.write(chars)
                f_trans.write("\n")

    return total_loss[0] / len(data_source), all_accuracies/bcnt


def evaluate_generator(whichdecoder, noise, epoch):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise)
    max_indices = \
        autoencoder.generate(whichdecoder, fake_hidden, maxlen=50, sample=args.sample)

    with open("%s/%s_generated%d.txt" % (args.outf, epoch, whichdecoder), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")


def train_ae(whichdecoder, batch, total_loss_ae, start_time, i):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
    output = autoencoder(whichdecoder, source, lengths, noise=True)
    flat_output = output.view(-1, ntokens)
    masked_output = flat_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion_ce(masked_output/args.temp, masked_target)
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    total_loss_ae += loss.data

    accuracy = None
    if i % args.log_interval == 0 and i > 0:
        probs = F.softmax(masked_output, dim=-1)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]
        cur_loss = total_loss_ae[0] / args.log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
              .format(epoch, i, len(train1_data),
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss), accuracy))

        with open("{}/log.txt".format(args.outf), 'a') as f:
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                    format(epoch, i, len(train1_data),
                           elapsed * 1000 / args.log_interval,
                           cur_loss, math.exp(cur_loss), accuracy))

        total_loss_ae = 0
        start_time = time.time()

    return total_loss_ae, start_time


def train_gan_g():
    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)
    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden)
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def grad_hook(grad):
    return grad * args.grad_lambda


''' Steal from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py '''
def calc_gradient_penalty(netD, real_data, fake_data):
    bsz = real_data.size(0)
    alpha = torch.rand(bsz, 1)
    alpha = alpha.expand(bsz, real_data.size(1))  # only works for 2D XXX
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gan_gp_lambda
    return gradient_penalty


def train_gan_d(whichdecoder, batch):
    # clamp parameters to a cube
    #for p in gan_disc.parameters():
    #    p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    gan_disc.train()
    optimizer_gan_d.zero_grad()

    # positive samples ----------------------------
    # generate real codes
    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # batch_size x nhidden
    real_hidden = autoencoder(whichdecoder, source, lengths, noise=False, encode_only=True)

    # loss / backprop
    errD_real = gan_disc(real_hidden)
    errD_real.backward(one)

    # negative samples ----------------------------
    # generate fake codes
    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    # loss / backprop
    fake_hidden = gan_gen(noise)
    errD_fake = gan_disc(fake_hidden.detach())
    errD_fake.backward(mone)

    # gradient penalty
    gradient_penalty = calc_gradient_penalty(gan_disc, real_hidden.data, fake_hidden.data)
    gradient_penalty.backward()

    optimizer_gan_d.step()
    errD = -(errD_real - errD_fake)

    return errD, errD_real, errD_fake


def train_gan_d_into_ae(whichdecoder, batch):
    ## clamp parameters to a cube
    #for p in gan_disc.parameters():
    #    p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))
    real_hidden = autoencoder(whichdecoder, source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(grad_hook)
    errD_real = gan_disc(real_hidden)
    errD_real.backward(mone)
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    optimizer_ae.step()

    return errD_real


print("Training...")
with open("{}/log.txt".format(args.outf), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

fixed_noise = to_gpu(args.cuda,
                     Variable(torch.ones(args.batch_size, args.z_size)))
fixed_noise.data.normal_(0, 1)
one = to_gpu(args.cuda, torch.FloatTensor([1]))
mone = one * -1

for epoch in range(1, args.epochs+1):
    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        with open("{}/log.txt".format(args.outf), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                    format(niter_gan))

    total_loss_ae1 = 0
    total_loss_ae2 = 0
    classify_loss = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_global = 1

    # loop through all batches in training data
    while niter < len(train1_data) and niter < len(train2_data):

        # train autoencoder ----------------------------
        for i in range(args.niters_ae):
            if niter == len(train1_data):
                break  # end of epoch
            total_loss_ae1, start_time = \
                train_ae(1, train1_data[niter], total_loss_ae1, start_time, niter)
            total_loss_ae2, _ = \
                train_ae(2, train2_data[niter], total_loss_ae2, start_time, niter)
            
            # train classifier ----------------------------
            classify_loss1, classify_acc1 = train_classifier(1, train1_data[niter])
            classify_loss2, classify_acc2 = train_classifier(2, train2_data[niter])
            classify_loss = (classify_loss1 + classify_loss2) / 2
            classify_acc = (classify_acc1 + classify_acc2) / 2
            # reverse to autoencoder
            classifier_regularize(1, train1_data[niter])
            classifier_regularize(2, train2_data[niter])

            niter += 1

        # train gan ----------------------------------
        for k in range(niter_gan):

            # train discriminator/critic
            for i in range(args.niters_gan_d):
                # feed a seen sample within this epoch; good for early training
                if i % 2 == 0:
                    batch = train1_data[random.randint(0, len(train1_data)-1)]
                    whichdecoder = 1
                else:
                    batch = train2_data[random.randint(0, len(train2_data)-1)]
                    whichdecoder = 2
                errD, errD_real, errD_fake = train_gan_d(whichdecoder, batch)

            # train generator
            for i in range(args.niters_gan_g):
                errG = train_gan_g()

            # train autoencoder from d
            for i in range(args.niters_gan_ae):
                if i % 2 == 0:
                    batch = train1_data[random.randint(0, len(train1_data)-1)]
                    whichdecoder = 1
                else:
                    batch = train2_data[random.randint(0, len(train2_data)-1)]
                    whichdecoder = 2
                errD_ = train_gan_d_into_ae(whichdecoder, batch)

        niter_global += 1
        if niter_global % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f (Loss_D_real: %.4f '
                  'Loss_D_fake: %.4f) Loss_G: %.4f'
                  % (epoch, args.epochs, niter, len(train1_data),
                     errD.data[0], errD_real.data[0],
                     errD_fake.data[0], errG.data[0]))
            print("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
                    classify_loss, classify_acc))
            with open("{}/log.txt".format(args.outf), 'a') as f:
                f.write('[%d/%d][%d/%d] Loss_D: %.4f (Loss_D_real: %.4f '
                        'Loss_D_fake: %.4f) Loss_G: %.4f\n'
                        % (epoch, args.epochs, niter, len(train1_data),
                           errD.data[0], errD_real.data[0],
                           errD_fake.data[0], errG.data[0]))
                f.write("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
                        classify_loss, classify_acc))

            # exponentially decaying noise on autoencoder
            autoencoder.noise_r = \
                autoencoder.noise_r*args.noise_anneal


    # end of epoch ----------------------------
    # evaluation
    test_loss, accuracy = evaluate_autoencoder(1, test1_data[:1000], epoch)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f}'.
          format(epoch, (time.time() - epoch_start_time),
                 test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)
    with open("{}/log.txt".format(args.outf), 'a') as f:
        f.write('-' * 89)
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                       test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')
    
    test_loss, accuracy = evaluate_autoencoder(2, test2_data[:1000], epoch)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f}'.
          format(epoch, (time.time() - epoch_start_time),
                 test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)
    with open("{}/log.txt".format(args.outf), 'a') as f:
        f.write('-' * 89)
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                       test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')

    evaluate_generator(1, fixed_noise, "end_of_epoch_{}".format(epoch))
    evaluate_generator(2, fixed_noise, "end_of_epoch_{}".format(epoch))

    if args.debug:
        continue

    # shuffle between epochs
    train1_data = batchify(corpus.data['train1'], args.batch_size, shuffle=True)
    train2_data = batchify(corpus.data['train2'], args.batch_size, shuffle=True)
    
    
test_loss, accuracy = evaluate_autoencoder(1, test1_data, epoch+1)
print('-' * 89)
print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
      'test ppl {:5.2f} | acc {:3.3f}'.
      format(epoch, (time.time() - epoch_start_time),
             test_loss, math.exp(test_loss), accuracy))
print('-' * 89)
with open("{}/log.txt".format(args.outf), 'a') as f:
    f.write('-' * 89)
    f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
            ' test ppl {:5.2f} | acc {:3.3f}\n'.
            format(epoch, (time.time() - epoch_start_time),
                   test_loss, math.exp(test_loss), accuracy))
    f.write('-' * 89)
    f.write('\n')

test_loss, accuracy = evaluate_autoencoder(2, test2_data, epoch+1)
print('-' * 89)
print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
      'test ppl {:5.2f} | acc {:3.3f}'.
      format(epoch, (time.time() - epoch_start_time),
             test_loss, math.exp(test_loss), accuracy))
print('-' * 89)
with open("{}/log.txt".format(args.outf), 'a') as f:
    f.write('-' * 89)
    f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
            ' test ppl {:5.2f} | acc {:3.3f}\n'.
            format(epoch, (time.time() - epoch_start_time),
                   test_loss, math.exp(test_loss), accuracy))
    f.write('-' * 89)
    f.write('\n')
