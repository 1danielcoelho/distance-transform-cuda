# distance-transform-cuda
CUDA implementation of Meijster's parallel algorithm for calculating the distance transform of a 2D image

![input](https://raw.githubusercontent.com/1danielcoelho/raytracing-rust/master/input.png) ![output](https://raw.githubusercontent.com/1danielcoelho/raytracing-rust/master/output.png)

Uses CUDA v9.1.

This started out as an attempt to implement "A Fast CUDA-based Implementation for the Euclidean Distance Transform" by Zampirolli and Filipe, as that is the top result when searching for CUDA and distance transforms.

I could not get their sample codes to run or produce anything, so I have modified the `edt_cols` kernel for a simpler procedure. The `edt_rows` kernel has been written based on equation (1) of Meijster's "A General Algorithm for Computing Distance Transforms In Linear Time". 

The performance is very similar to what Zampirolli and Filipe claim: My NVIDIA GTX 1060 6GB is capable of calculating the distance transform of a 256x256 image in 3ms with this kernel. Zampirolli and Filipe claim their kernel completes, for the same image, in about 1.7ms with an NVIDIA Titan X.

# Installation
* Easiest way is to duplicate the "0_Simple/template" Cuda sample project and just replace all the code with the single .cu file in this repository

# TODO
* I have not tried to optimize this extremely hard, as the performance is sufficient for me right now. I believe the `edt_cols` kernel can be further broken down, having multiple threads per column
