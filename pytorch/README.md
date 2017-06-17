# PyTorch ARAE for Text Generation

### Requirements
- PyTorch, JSON, Argparse
- (Optional) KenLM (https://github.com/kpu/kenlm)

## Pretrained Version

1) Download and unzip https://drive.google.com/drive/folders/0B4IZ6lmAKTWJSE9UNFYzUkphaVU?usp=sharing.

2) Run: 

    `python generate.py --load_path ./maxlen30`

    (Requires CUDA and Python)

## Example Generations

### Sentence Generations
#### Maximum Length 15
```
man catches kids playing the electric cello with many colors .
Three boys in bathing suits on small mountain covered steps
People and part are competing through the band , are crossing .
The young people were sitting at a train after the art building .
people are dressed to play soccer .
The man is using <oov> as a cop passes .
People look gathering to the park at the bar
The little blonde woman was walking across a car to catch .
Group of kids hold a metal jump rope
Three kids are in a van .
The woman is jumping
The <oov> woman have their <oov> to over the street beside pigeons .
A man dressed <oov> of a foreign country .
This girl in dark shorts was cooking hot
The janitor is resting outside .
Four men and lady with bags were at a truck .
A young man is at a computer desk .
Some men are standing on a couch .
There are two musicians on a white table .
```

#### Maximum Length 30
```
The man is driving a car on the pool .
The child is behind the tree .
A basketball team wearing robes runs .
A yellow car carries a white mountain .
There are many people playing at the arcade this class .
A man playing fetch with ice <oov> .
A group of woman is pushing the wagon in a crowded desert location .
A man in a group of woman is standing on a platform in a plain pair of shorts .
A man is in a white shirt carrying a yellow bag and a small bird .
The man is relaxing .
A middle-aged woman with pigtails playing a violin in the street before her life .
There is three men in a tent on stage .
a young man is do handstands .
The girl is a <oov> rollerblading .
A group of people are enjoying the same
A woman is play a game with another friend .
A homeless woman posing for a children with glasses inside a kitchen .
The hockey players talking for fun .
A caucasian boy wearing an orange suit is looking at a beautiful city .
```

### Z-space Interpolations
#### Maximum Length 15
```
A lady is juggling with a cigarette while <oov> in the background has an umbrella
A couple is setting up on their stage and an <oov> is near .
A woman is relaxing on stage and holding the next of public .
A woman is on top of some People .
There are people on top of a staircase .
```

```
A man napping , .
A human lying .
A band swings on .
A young family with balloons .
An audience plays music with balloons .
```

```
The men are discussing the football for the church .
The men are reading the magazine on the church .
The woman are reading on the same the Chinese church .
The woman are reading a room in an outdoor auditorium .
Children taking a photo is playing with large books .
```

#### Maximum Length 30
```
A girl in an red dress is placing her friend and standing on the bleachers performing surgery .
A girl in black shirt is showing her parents , and kids on a romantic date .
A girl in black shirt is getting her parents , watching a car .
A man in a black shirt is getting ready and a friend enjoys .
A man in a black shirt is getting a friends family having a park .
```

```
The man is eating in the park
The man is walking an home
The group walks to the water .
Guys are near them .
Men are watching people .
```

```
A little guy is young adult playing .
A little cowboy girl is playing boys .
A happy man is driving something in his hair .
A group of <oov> walks is wearing sunglasses in a white box .
A group of <oov> woman is putting people in a white shop with the window .
```


## Data Preparation

### SNLI Data Preparation
- Download dataset and unzip: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
- Run `python snli_preprocessing.py --in_path PATH_TO_SNLI --out_path PATH_TO_PROCESSED_DATA`
    - Example: `python snli_preprocessing.py --in_path ../Data/snli_1.0 --out_path ../Data/snli_lm`
    - The script will create the output directory if it doesn't already exist
- For more information on SNLI see: https://nlp.stanford.edu/projects/snli/

### Your Customized Datasets
If you would like to train a text ARAE on another dataset, simply
1) Create a data directory with a `train.txt` and `test.txt` files with line delimited sentences.
2) Run training command with the `--data_path` argument pointing to that data directory.

## Train
1) To train without KenLM: 

    `python train.py --data_path PATH_TO_PROCESSED_DATA --cuda --no_earlystopping`

2) To train with KenLM for early stopping: 

    `python train.py --data_path PATH_TO_PROCESSED_DATA --cuda --kenlm_path PATH_TO_KENLM_DIRECTORY`

- When training on default parameters the training script will output the logs, generations, and saved models to: `./output/example`

### Model Details
- We train on sentences that have up to 30 tokens and take the most likely word (argmax) when decoding (there is an option to sample when decoding as well).
- For a numerical way for early stopping, after the model has trained for a specified minimum number of epochs, we periodically train a n-gram language model (with modified Kneser-Ney and Laplacian smoothing) on 100,000 generated sentences and evaluate the perplexity of real sentences from a held-out test set. If the perplexity does not improve over that of the lowest perplexity seen for a certain number of iterations (patience), we end training.


### KenLM Installation:
- Download stable release and unzip: http://kheafield.com/code/kenlm.tar.gz
- Need Boost >= 1.42.0 and bjam
    - Ubuntu: `sudo apt-get install libboost-all-dev`
    - Mac: `brew install boost; brew install bjam`
- Run within kenlm directory:
    ```bash
    mkdir -p build
    cd build
    cmake ..
    make -j 4
    ```
- `pip install https://github.com/kpu/kenlm/archive/master.zip`
- For more information on KenLM see: https://github.com/kpu/kenlm and http://kheafield.com/code/kenlm/



