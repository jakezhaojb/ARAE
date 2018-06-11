# ARAE for Language generator

### Requirements
- PyTorch, JSON, Argparse
- KenLM (https://github.com/kpu/kenlm)

## Pretrained version

1) Download and unzip from [here](https://drive.google.com/file/d/1h66T8UdFuNWWjvmLLcYC9bHExpNv8NR4/view?usp=sharing)

2) Run: 

```
python generate.py --load_path ./snli_pretrained
```


## Example generations

```
young boy bowling to the death .
a woman wearing sunglasses for a woman to talk .
a man is laying down enjoying a mural of the leaning against his eyes .
animals are reading from old .
the women are n't admiring anything .
the man is watching the beach on his phone .
a woman is eating .
a shirtless girl carrying a backpack .
the couple is holding on the beach .
two women face off a taxi .
```

## Examplar interpolations

```
the weather does winter .
the two humans repairing .
two people ride a bus .
two people standing inside a firetruck .
two men sitting down on a stage posing in a room .

a man has two things up .
a woman has three green .
a woman takes a smoke outside .
a woman takes a bath outside .
a woman takes a bath with her hands .

an outdoor food cart .
a kid outside is fixing .
a man is holding a painting .
a man is on a scooter
a dog on a leash is in a car .

a man on a hill
the dog on the shore
the dog is on the stairs .
the car is advertising .
the car is made surgery .

a man in blue glasses has paint on the wall of lunch .
a boy is playing with toys on the train tracks .
a boy is looking at clothes on display .
a child waits to hold on a shovel to stop .
a child smiles to someone not by a river
```

## Preparation

### SNLI Data Preparation
- Download dataset and unzip:
    
    mkdir data; cd data
    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip snli_1.0.zip
    cd ..; python snli_preprocessing.py --in_path data/snli_1.0 --out_path data/snli_lm

### Your Customized Datasets
If you would like to train a text ARAE on another dataset, simply
1) Create a data directory with a `train.txt` and `test.txt` files with line delimited sentences.
2) Run training command with the `--data_path` argument pointing to that data directory.

### KenLM Installation:
- Download stable release and unzip: http://kheafield.com/code/kenlm.tar.gz
- Need Boost >= 1.42.0 and bjam
    - Ubuntu: `sudo apt-get install libboost-all-dev`
    - Mac: `brew install boost; brew install bjam`
- Run *within* kenlm directory:
    ```bash
    mkdir -p build
    cd build
    cmake ..
    make -j 4
    ```
- `pip install https://github.com/kpu/kenlm/archive/master.zip`
- For more information on KenLM see: https://github.com/kpu/kenlm and http://kheafield.com/code/kenlm/


## Train

```
python run_snli.py --data_path ./data/snli_lm --no_earlystopping
```


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
