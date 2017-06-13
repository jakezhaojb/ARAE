import os
import json
import codecs
import argparse

"""
Transforms SNLI data into lines of text files
    (data format required for ARAE model).
Gets rid of repeated premise sentences.
"""


def transform_data(in_path):
    print("Loading", in_path)

    premises = []
    hypotheses = []

    last_premise = None
    with codecs.open(in_path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)

            # load premise
            raw_premise = loaded_example['sentence1_binary_parse'].split(" ")
            premise_words = []
            # loop through words of premise binary parse
            for word in raw_premise:
                # don't add parse brackets
                if word != "(" and word != ")":
                    premise_words.append(word)
            premise = " ".join(premise_words)

            # load hypothesis
            raw_hypothesis = \
                loaded_example['sentence2_binary_parse'].split(" ")
            hypothesis_words = []
            for word in raw_hypothesis:
                if word != "(" and word != ")":
                    hypothesis_words.append(word)
            hypothesis = " ".join(hypothesis_words)

            # make sure to not repeat premiess
            if premise != last_premise:
                premises.append(premise)
            hypotheses.append(hypothesis)

            last_premise = premise

    return premises, hypotheses


def write_sentences(write_path, premises, hypotheses, append=False):
    print("Writing to {}\n".format(write_path))
    if append:
        with open(write_path, "a") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')
    else:
        with open(write_path, "w") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default="../Data/snli_1.0",
                        help='path to snli data')
    parser.add_argument('--out_path', type=str, default="../Data/snli_lm",
                        help='path to write snli language modeling data to')
    args = parser.parse_args()

    # make out-path directory if it doesn't exist
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print("Creating directory "+args.out_path)

    # process and write test.txt and train.txt files
    premises, hypotheses = \
        transform_data(os.path.join(args.in_path, "snli_1.0_test.jsonl"))
    write_sentences(write_path=os.path.join(args.out_path, "test.txt"),
                    premises=premises, hypotheses=hypotheses)

    premises, hypotheses = \
        transform_data(os.path.join(args.in_path, "snli_1.0_train.jsonl"))
    write_sentences(write_path=os.path.join(args.out_path, "train.txt"),
                    premises=premises, hypotheses=hypotheses)

    premises, hypotheses = \
        transform_data(os.path.join(args.in_path, "snli_1.0_dev.jsonl"))
    write_sentences(write_path=os.path.join(args.out_path, "train.txt"),
                    premises=premises, hypotheses=hypotheses, append=True)
