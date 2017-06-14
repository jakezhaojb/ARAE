import numpy as np
import pickle
from collections import defaultdict
import spacy
from spacy.symbols import nsubj, VERB
from models import load_models, generate
import argparse
import torch

nlp = spacy.load("en")


def get_subj_verb(sent):
    "Given a parsed sentence, find subject, verb, and subject modifiers."
    sub = set()
    verbs = set()
    mod = set()
    for i, possible_subject in enumerate(sent):
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            if possible_subject.head.head.pos == VERB:
                verbs.add(str(possible_subject.head.head))
            else:
                verbs.add(str(possible_subject.head))
            sub.add(str(possible_subject))
            c = list(possible_subject.children)
            for w in c:
                mod.add(str(w))
            if not c:
                mod.add(str(sent[i-1]) if i != 0 else "NONE")
    return verbs, sub, mod


def featurize(sent):
    "Given a sentence construct a feature rep"
    verb, sub, mod = get_subj_verb(sent)
    d = {}

    def add(d, pre, ls):
        for l in ls:
            d[pre + "_" + l] = 1
    add(d, "VERB", list(verb)[:1])
    add(d, "MOD", list(mod))
    add(d, "NOUN", list(sub)[:1])
    return d


def gen(vec):
    "Generate argmax sentence from vector."
    return generate(autoencoder, gan_gen, z=torch.FloatTensor(vec).view(1, -1),
                    vocab=idx2word, sample=False,
                    maxlen=model_args['maxlen'])


def gen_samples(vec):
    "Generate sample sentences from vector."
    sentences = []
    sentences = generate(autoencoder, gan_gen, z=torch.FloatTensor(vec)
                         .view(1, -1).expand(20, vec.shape[0]),
                         vocab=idx2word, sample=True,
                         maxlen=model_args['maxlen'])[0]
    return sentences


def switch(vec, mat, rev, f1, f2):
    "Update vec away from feature1 and towards feature2."
    means = []
    m2 = np.mean(mat[list(rev[f2])], axis=0)
    for f in f1:
        if list(rev[f]):
            means.append(np.mean(mat[list(rev[f])], axis=0))
    m1 = np.mean(means) if f1 else np.zeros(m2.shape)

    val = vec + (m2 - m1)
    return val, vec - m1


def alter(args):
    sents, features, rev, mat = pickle.load(open(args.dump, "br"))
    mat = mat.numpy()

    # Find examples to alter toward new feat.
    new_feat = args.alter

    pre = new_feat.split("_")[0]
    word = new_feat.split("_")[1]
    for i in range(args.nsent):
        vec = mat[i]

        for j in range(10):
            sent = gen(vec)[0]
            f = featurize(nlp(sent))
            print("Sent ", j, ": \t ", sent, "\t")
            if word in sent:
                break

            # Compute the feature distribution associated with this point.
            samples = gen_samples(vec)
            feats = [f] * 50
            for s in samples:
                feats.append(featurize(nlp(s)))

            mod = []
            for feat in feats:
                for feature in feat:
                    if feature.startswith(pre):
                        mod.append(feature)

            # Try to updated the vector towards new_feat
            update, temp = switch(vec, mat, rev, mod, new_feat)
            if j == 0:
                orig = temp

            # Interpolate with original.
            vec = 0.2 * orig + 0.8 * update

        print()
        print()


def dump_samples(args):
    "Construct a large number of samples with features and dump to file."
    all_features = []
    all_sents = []

    batches = args.nbatches
    batch = args.batch_size
    samples = 1
    total = batches * batch * samples
    all_zs = torch.FloatTensor(total, model_args['z_size'])
    rev = defaultdict(set)

    for j in range(batches):
        print("%d / %d batches " % (j, batches))
        noise = torch.ones(batch, model_args['z_size'])
        noise.normal_()
        noise = noise.view(batch, 1, model_args['z_size'])\
                     .expand(batch, samples,
                             model_args['z_size']).contiguous()\
                     .view(batch*samples,
                           model_args['z_size'])
        sentences = generate(autoencoder, gan_gen, z=noise,
                             vocab=idx2word, sample=True,
                             maxlen=model_args['maxlen'])

        for i in range(batch * samples):
            k = len(all_features)
            nlp_sent = nlp(sentences[i])
            feats = featurize(nlp_sent)
            all_sents.append(sentences[i])
            all_features.append(feats)
            for f in feats:
                rev[f].add(k)
            all_zs[k] = noise[i]
    pickle.dump((all_sents, all_features, rev, all_zs), open(args.dump, "bw"))


def main(args):
    if args.mode == 'gen':
        dump_samples(args)
    elif args.mode == 'alter':
        alter(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch experiment')
    parser.add_argument('mode', default='gen',
                        help='choices [gen, alter]')
    parser.add_argument('--load_path', type=str,
                        help='directory to load models from')

    parser.add_argument('--dump', type=str, default="features.pkl",
                        help='path to sample dump')
    parser.add_argument('--nbatches', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--alter', type=str, default="")
    parser.add_argument('--nsent', type=int, default=100)
    args = parser.parse_args()
    model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.load_path)

    main(args)
