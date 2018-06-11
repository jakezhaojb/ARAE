# ARAE
Code for the paper "Adversarially Regularized Autoencoders (ICML 2018)" by Zhao, Kim, Zhang, Rush and LeCun https://arxiv.org/abs/1706.04223


## Disclaimer
Major updates on 06/11/2018:
* WGAN-GP replaced WGAN
* added 1BWord dataset experiment
* added Yelp transfer experiment
* removed unnecessary tricks
* added both RNNLM and ngram-LM evaluation for both forward and reverse PPL.

## File structure
* lang: ARAE for language generation, on both 1B word benchmark and SNLI
* yelp: ARAE for yelp style transfer
* mnist (in Torch): ARAE for discretized MNIST


## Reference

```
@ARTICLE{2017arXiv170604223J,
   author = {{Junbo} and {Zhao} and {Kim}, Y. and {Zhang}, K. and {Rush}, A.~M. and 
	{LeCun}, Y.},
    title = "{Adversarially Regularized Autoencoders for Generating Discrete Structures}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1706.04223},
 primaryClass = "cs.LG",
 keywords = {Computer Science - Learning, Computer Science - Computation and Language, Computer Science - Neural and Evolutionary Computing},
     year = 2017,
    month = jun,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv170604223J},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```




