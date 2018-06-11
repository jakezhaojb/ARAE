# ARAE for Language generator

## Requirements
- PyTorch, JSON, Argparse
- KenLM (https://github.com/kpu/kenlm)

## Train and Pretrain models
* [SNLI](doc/README_snli.md)
* [1BWord benchmark](doc/README_oneb.md) 

### Your Customized Datasets
If you would like to train a text ARAE on another dataset, simply
1) Create a data directory with a `train.txt` and `test.txt` files with line delimited sentences.
2) Run training command with the `--data_path` argument pointing to that data directory.

## Evaluation with RNNLM

To evaluate the reverse PPL with an RNNLM, first preprocess the data with the generated/real text files, e.g.

```
python preprocess_lm.py --trainfile generated-data.txt --valfile real-val.txt --testfile real-test.txt
```

To train the model

```
python train_rnnlm.py --train_file lm-data-train.hdf5 --val_file lm-data-val.hdf5 --checkpoint_path lm-model.ptb
```

To evaluate on test

```
python train_rnnlm.py --trainfile lm-data-train.hdf5 --val_file lm-data-test.hdf5 --train_from lm-model.ptb --test 1 
```



