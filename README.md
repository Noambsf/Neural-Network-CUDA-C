# Neural Network MNIST Training in CUDA C

This repository contains a CUDA C implementation of a Fully Dense Neural Network trained and evaluated on MNIST from scratch in C optimized with CUDA. 

## Requirements

Before running the code, ensure you have the required dependencies installed. You can install them using:

```bash
sudo apt install nvidia-cuda-toolkit
```

## Running the code

First compile using :

```bash
nvcc -o ./ann.out main.cu matrix.cu ann.cu mnist.cu -lm
```

Then you can run the training with :

```bash
./ann.out
```

## References

This CUDA optimization project was conducted with **Baptiste BOUTAUD** as part of the end-of-course project for the course **"Programming on Graphics Processor"** at Ecole Centrale de Nantes taught by **Pierre Emmanuel Hladik**.

You can find the original code (written in C only) that we used as a baseline at the following link:

[Original Code Repository](https://gitlab.univ-nantes.fr/hladik-pe-1/ecn-gpu-tp/-/tree/main/TP1/code_TP?ref_type=heads)

## Final results

In conclusion, our CUDA optimization pipeline reduced the average training time per epoch from approximately **53.2 seconds** to **9.1 seconds**. All experiments were conducted on the same **Jetson Nano** provided by the school.
