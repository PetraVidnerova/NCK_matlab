# MATLAB CUDA/MEX code extensions
The repo contains refactored and compiled code implementing GANs using MATLAB GPU Coder.
The code is provided in form of *.m scripts that implement GANs and compiled versions
in form of *.MEX files. Compiled versions directly utilize CUDA kernels to speed up computation.

## Authors
D. Coufal, F. Hakl, P. Vidnerová. The Czech Academy of Sciences, Institute of Computer Science

## Keywords
MATLAB GPU Coder, CUDA, generative adversial networks 

## Requirements
MATLAB, MATLAB GPU Coder

## Main features:
- CUDA compiled code
- speeding up generation phase of GANs

## List of scripts
```
- nck_dcgan.m, nck_lsgan - GANs implemented in MATLAB, parameters are set within the scripts
- *_generate.m - functions to generate rxc images from gan, r,c are parameters
- *_generate_mex.mexw64 - compiled versions for r=100, c_100; generate 10000 images
```

## Speed of generation on RTX 2080 Ti
for uncompiled, CPU version generation of 10k images takes about 10 sec
```
tic;*_generate;toc
```
for compiled, CUDA native version, generation of 10k images takes about 1 sec
```
tic;*_generate_mex;toc
```
Consider to run the above commands at least twice to get stable times.
The first run is usually not reliable.

## Acknowledgement
This work was partially supported by the TAČR grant TN01000024 and institutional support
of the Institute of Computer Science RVO 67985807