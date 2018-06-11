## Pretrained version TODO

1) Download and unzip from [here]()

2) Run: 

```
python generate.py --load_path ./oneb_pretrained -steps 8
```


## Example generations

```
the store collapsed after two days , but the situation is turning out as this play .
you meet i with myself pushing a <oov> <oov> . the best solution to that type and mental illness .
the french worker pulled a gas leak at the group , whose eyes shot found .
quite recently held as " one bit . "
the church of manchester , a known member of merseyside parish , claims crime .
the couple will have to flood three wards with the third estate , <oov> of course .
it was not giving " hope " against <oov> felt and was modest .
the plane was a public marketing .
there are the unfortunately consequences .
the market to keep consumers below the ranks of around 300 , " she never said ."
```

## Examplar interpolations

```
it said colorado with its men who .
it said last may of " may remember . "
" it sounds when a lot of refugees may say the word .
it <oov> me what i said , what we told the yankees on new tests .
the three <oov> said it 's about " who in the pub of america , " he said .
the three said teachers <oov> about the other 90 minutes with new signs of problems , he said .
why if anyone could be allowed to enter the nation 's most comfortable university of its worst ?
why pay millions of the nurses to marry with a job , and don 't think of public life .

the exhibition followed the set panel to carry its staff to finish their lives about their work hours .
the exhibition followed the set panel with one drink ; half per cent said they were prepared to retire .
the publisher said it believes the offences may be worth <oov> in <oov> , over eight years older .
the shares slid then traded highs , the euro with $ <oov> to settle at $ <oov> .
to be guided 's own offense is attempting to <oov> the problem over your professional league .
to be handled with paul 's length and <oov> <oov> constantly is the master sergeant .
he was able to convert his professional <oov> 's <oov> outfit with these words .
he said a formula could be <oov> : jeremy clarkson , 18 , always <oov> .

the bills signed him the candidate and rep. pat .
he then filed the bankruptcy and found friday 5 .
he estimates 100 other the 33 million died in 2007 .
500 feet far near the bottom a & p .
oil ceo will have the milwaukee brewers '
oil eventually will continue the last session above 10,000 .
the vast majority will have died last week , the youngest day .
authorities have obviously blamed a policeman holding a pair of the rocks .

he still , it matters .
he still looks like it , " ali said .
he failed but , she said two errors .
such , short term " fundamentally broken old . "
such a key idea is strictly , " he asked .
he finds this : they sell and destroy the ford body , which they do .
he threw it from now the taliban but virtually so without command and military equipment .
in the opposite mix it is derived from a modern television , and it caught in all worlds .

if banks do not walk out , the process was able to drive without running , he said .
to work , they must roll up the distance to get through with something nice walking .
to make " the big cat " games , people finally get over to all profit .
but <oov> had not been seriously the problem - finishing duke and with <oov> boards .
but <oov> , and the us company boss <oov> <oov> , " csi " between january .
but i was in <oov> <oov> near the la times during an <oov> rather than in 2007 .
but , <oov> 's <oov> grant <oov> <oov> the korean peninsula back across <oov> .
in 1977 , <oov> <oov> 's christian parish appealed the letter into <oov> <oov> ."'"
```


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
python run_oneb.py --data_path ./data/oneb --no_earlystopping
```
