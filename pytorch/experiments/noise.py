import argparse
from models import load_models, generate
import torch
import difflib
import numpy.linalg

ENDC = '\033[0m'
BOLD = '\033[1m'


def main(args):
    noise = torch.ones(100, model_args['z_size'])

    for k in range(10):
        noise[0].normal_()
        for i in range(1, 100):
            noise[i].normal_()
            noise[i] = noise[i] / (10 * numpy.linalg.norm(noise[i]))
            noise[i] += noise[0]
        sents = gen(noise)
        print(sents[0])
        seen = set()
        seen.add(sents[0])
        for i in range(40):
            if sents[i] not in seen:
                seen.add(sents[i])
                a = sents[0].split()
                b = sents[i].split()
                sm = difflib.SequenceMatcher(a=a, b=b)

                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == "equal":
                        print(" ".join(b[j1:j2]), end=" ")
                    if tag == "replace":
                        print(BOLD + " ".join(b[j1:j2]) + ENDC, end=" ")
                        # print("*" + " ".join(b[j1:j2]) + "*", end=" ")
                print()
        print()

def gen(vec):
    "Generate argmax sentence from vector."
    return generate(autoencoder, gan_gen, z=vec,
                    vocab=idx2word, sample=False,
                    maxlen=model_args['maxlen'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch experiment')
    parser.add_argument('--load_path', type=str,
                        help='directory to load models from')
    args = parser.parse_args()
    model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.load_path)

    main(args)
