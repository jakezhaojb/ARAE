## Language generation

Torch implementation for SNLI language generation experiment.

### Requirements
This code runs using torch. It is only tested on GPU. The following torch libraries are required:
nn, cunn, cudnn, optim, image, nngraph, hdf5


### SNLI dataset preparation


### Train
```
th main.lua
```

The output folder: `${save_name}`, with structure
```
.
+-- ${save_name}
  +-- model_${epoch}.t7
  +-- ${epoch}.log
```
