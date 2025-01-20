#ifndef ANNC_H
#define ANNC_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.h"

typedef struct 
{
    unsigned minibatch_size;
    unsigned number_of_neurons;
    
    matrix_t* weights;
    matrix_t* biases;

    matrix_t* z;
    matrix_t* activations;
    
    matrix_t* delta;
} layer_t;

typedef struct 
{
    void (*f)(double*, double*, unsigned, unsigned);
    void (*fd)(double*, double*, unsigned, unsigned);
    double alpha;
    unsigned minibatch_size;
    unsigned input_size;
    unsigned number_of_layers;
    layer_t** layers;
} ann_t;

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer);

layer_t * create_layer(unsigned l, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size);

void set_input(ann_t *nn, matrix_t* input);

void print_nn(ann_t *nn);

void forward_GPU(ann_t *nn, double (*activation_function)(double));

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double));

void backward_gpu(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double));

void backward_gpu_cpu(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double));

#endif