//modifié

#include "ann.h"
#include "matrix.h"
#include "error.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

#define TILE_WIDTH 8

// __global__
// void matrix_dot_GPU( 
//     //première version du kernel de multiplication matricielle 
//     // inutilisée par la suite 
//    double *A, double *B, double *C,
//    int numARows, int numAColumns,
//    int numBRows, int numBColumns
// )
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x; 


//     if (row < numARows && col < numBColumns)
//     {
//         double sum = 0;
//         for (int ii = 0; ii < numAColumns; ii++)
//         {
//             sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
//         }
//         C[row * numBColumns + col] = sum;
//     }
// }

// __global__
// void matrix_dot_GPU_1D( 
//     // version du kernel de multiplication matricielle  pour l'opération Matrice * One 
//     // performances améliorées par rapport à matrxi_dot_GPU dans ce cas d'utilisation
//     // inutilisée par la suite 
//    double *A, double *C,
//    int numARows, int numBColumns
// )
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x; 
    


//     if (row < numARows && col < numBColumns)
//     {
//         C[row * numBColumns + col] = A[row];
//     }
// }

__global__ 
void matrix_dot_shared(double *m1, double *m2, double *res, unsigned m1_r, unsigned m1_c, unsigned m2_r, unsigned m2_c) 
{
    // Version du kernel de multiplication matricielle qui permet des réaliser 
    // les deux produits matriciels simultanément en un seul kernel. Et ainsi, la Synchronization
    // Les performances étaient améliorées
    // Mais inutilisée au profit de : forward_gpu_z
    __shared__ double ds_m1[TILE_WIDTH*TILE_WIDTH];
    __shared__ double ds_m2[TILE_WIDTH*TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = tx + blockIdx.x * TILE_WIDTH;
    int col = ty + blockIdx.y * TILE_WIDTH;

    double pvalue = 0;

    for (int m = 0; m < (m1_c - 1) / TILE_WIDTH + 1; m++) {
        if (row < m1_r && (m * TILE_WIDTH) + ty < m1_c) {
            ds_m1[ty + (tx*TILE_WIDTH)] = m1[(m*TILE_WIDTH) + ty + row * m1_c];
        } else {
            ds_m1[ty + (tx*TILE_WIDTH)] = 0.0;
        }
        if (tx + (m * TILE_WIDTH) < m2_r && col < m2_c) {
            ds_m2[tx + (ty*TILE_WIDTH)] = m2[col +(tx + (m * TILE_WIDTH)) * m2_c];
        } else {
            ds_m2[tx + (ty*TILE_WIDTH)] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            pvalue = pvalue + (ds_m1[k + tx * TILE_WIDTH] * ds_m2[k + ty * TILE_WIDTH]);
        }
        __syncthreads();
    }

    if (row < m1_r && col < m2_c) {
        res[col + (row * m2_c)] = pvalue;
    }
}

__global__
void matrix_sum_act(
    // Version du kernel de l'addition matricielle 
    //Mais inutilisée au profit de kernels avancés comme : forward_gpu_z
double *m1, double *m2, double *res, 
unsigned int rows, unsigned int columns,
double (*f)(double)
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < columns)
    {
        int idx = row * columns + col;
        res[idx] = m1[idx] + m2[idx];
        res[idx] = f(res[idx]);
    }
}



__global__
void forward_gpu_z( 
    // Pour le calcul z = w(l)*a(l) + b(l)*one
    // Compacte l'ensemble du calcul en un seul appel kernel. 
    // Utilisée par la suite dans forward_GPU
   double *w, double *a,double *b, double *z,
   int numWRows, int numWColumns,
   int numARows, int numAColumns
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < numWRows && col < numAColumns)
    {
        double sum = 0;
        for (int ii = 0; ii < numWColumns; ii++)
        {
            sum += w[row * numWColumns + ii] * a[ii * numAColumns + col];
        }
        z[row * numAColumns + col] = sum + b[row];
    }
}


__global__
void backward_gpu_b( 
    // Pour le calcul b^l = b^l - alpha / m . delta^l x 1^T
    // Compacte l'ensemble du calcul en un seul appel kernel. 
    // Utilisée pour la backpropagation
   double *b, double *delta_l, double alpha,
   int numbRows, int minibatch_size
)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < numbRows)
    {
        double sum = 0;
        for (int ii = 0; ii < minibatch_size; ii++)
        {
            sum += delta_l[row * minibatch_size + ii]; 
        }
        b[row] -= - sum * alpha/minibatch_size ;
    }
}

__global__
void backward_gpu_w( 
    // Pour le calcul w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T
    // Compacte l'ensemble du calcul en un seul appel kernel. 
    // Utilisée pour la backpropagation
   double *w, double *delta_l, double *a, double alpha,
   int numDelta_lRows,int numARows, int minibatch_size
)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numDelta_lRows && col < numARows)
    {
        double sum = 0;
        for (int ii = 0; ii < minibatch_size; ii++)
        {
            sum += delta_l[row * minibatch_size + ii]*a[ii + col * minibatch_size] ;
        }
        w[row*numARows + col] -= sum * alpha/minibatch_size ;
    }
}

// on fait passer la fonction d'activation et sa dérivé en fonction __device__ afin qu'elles soient accessible depuis le GPU. 
// Ici on est dans un cas simple, car on a qu'une seule fonction d'activation, donc pas de problème
// mais dans un problème avce plusieurs fonctions d'activations, il faudrait faire des modifications. 
__device__
double sigmoid_device(double x)
{
    return 1 / (1 + exp(-x));
}

__device__
double dsigmoid_device(double x)
{
    return sigmoid_device(x)*(1-sigmoid_device(x));
}


__global__
void backward_gpu_delta_L(double *delta_l, double *a_l, double *y,
                              double *z_l, int numNeurons, int minibatch_size)
{
    // Pour le calcul b^l <- b^l - alpha /m . delta^l x (One)^T
    // Compacte l'ensemble du calcul en un seul appel kernel. 
    // Utilisée pour la backpropagation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numNeurons && col < minibatch_size)
    {
        int index = row * minibatch_size + col;

        delta_l[index] = (a_l[index] - y[index]) * dsigmoid_device(z_l[index]);
    }
}

__global__
void backward_gpu_hidden_layer(double *delta_l_minus_1, double *w_l, double *delta_l,
                                double *z_l_minus_1, int numNeurons_l_minus_1,
                                int numNeurons_l, int minibatch_size) 
{
    // Pour le calcul delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))
    // Compacte l'ensemble du calcul en un seul appel kernel. 
    // Utilisée pour la backpropagation

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numNeurons_l_minus_1 && col < minibatch_size)
    {
        int index = row * minibatch_size + col;

        double sum = 0.0;
        for (int ii = 0; ii < numNeurons_l; ii++)
        {
            sum += w_l[ii * numNeurons_l_minus_1 + row] * delta_l[ii * minibatch_size + col];
        }

        delta_l_minus_1[index] = sum * dsigmoid_device(z_l_minus_1[index]);
    }
}

double normalRand(double mu, double sigma)
{
    // Unchanged
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;
    bool generate = false;
    double z1 = 0.0;

	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    // Unchanged
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    // On a alloue la mémoire via Cuda en mémoire unifiée afin de pouvoir l'utiliser par la suite. 
    ann_t * nn ;
    

    cudaMallocManaged(&nn, sizeof(ann_t));
    cudaMallocManaged(&(nn->layers), sizeof(layer_t));
    
    
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }
    
    
    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    // On a alloue la mémoire via Cuda en mémoire unifiée afin de pouvoir l'utiliser ensuite. 

    layer_t *layer;
    cudaMallocManaged(&layer, sizeof(layer_t));
    


    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);
    
    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    //Inchangée
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    //Inchangée
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward_GPU(ann_t *nn, double (*activation_function)(double))
{
    unsigned numColumns = nn->minibatch_size;
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        unsigned numRows = nn->layers[l]->number_of_neurons;
        
        dim3 blockDim(8,8); // dimension de nos block 
        dim3 gridDim(ceil(float(numColumns)/blockDim.x) + 1, ceil(float(numRows)/blockDim.y) + 1);// dimension de notre grid 

        forward_gpu_z<<<gridDim,blockDim>>>(nn->layers[l]->weights->m, nn->layers[l-1]->activations->m,nn->layers[l]->biases->m , nn->layers[l]->z->m,
                                            nn->layers[l]->weights->rows,nn->layers[l]->weights->columns,nn->layers[l-1]->activations->rows,
                                            nn->layers[l-1]->activations->columns);        
        cudaDeviceSynchronize();
        
        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); 
    }
    
}


void backward_gpu(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    // unsigned numColumns = nn->minibatch_size;
    unsigned minibatch_size =nn->layers[0]->minibatch_size;
    unsigned L = nn->number_of_layers-1;
    dim3 blockDim(8,8);
    dim3 gridDim(ceil(float(nn->minibatch_size)/blockDim.x) + 1, ceil(float(nn->layers[L]->number_of_neurons)/blockDim.y) + 1);

    
    backward_gpu_delta_L<<<gridDim, blockDim>>>(nn->layers[L]->delta->m, nn->layers[L]->activations->m,
                                                   y->m, nn->layers[L]->z->m,
                                                   nn->layers[L]->number_of_neurons, minibatch_size);

    cudaDeviceSynchronize(); 

    for (int l = L; l > 1; l--)
    {

        unsigned numRows = nn->layers[l-1]->number_of_neurons;
        unsigned numColumns =nn->layers[l]->number_of_neurons;

        dim3 gridDim(ceil(float(nn->minibatch_size)/blockDim.x) + 1, ceil(float(numRows)/blockDim.y) + 1);


        backward_gpu_hidden_layer<<<gridDim, blockDim>>>(nn->layers[l-1]->delta->m, nn->layers[l]->weights->m,
                                                      nn->layers[l]->delta->m, nn->layers[l-1]->z->m,
                                                      numRows, numColumns, minibatch_size);

        cudaDeviceSynchronize();
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {

        unsigned numRows = nn->layers[l]->number_of_neurons;
        unsigned numCols = nn->layers[l-1]->number_of_neurons;
        

        dim3 gridDim(ceil(float(numCols/blockDim.x)) + 1, ceil(float(numRows/blockDim.y)) + 1);

        backward_gpu_w<<<gridDim, blockDim>>>(nn->layers[l]->weights->m,nn->layers[l]->delta->m,nn->layers[l-1]->activations->m,nn->alpha,numRows,numCols,minibatch_size);
        cudaDeviceSynchronize();  
        
        backward_gpu_b<<<ceil(float(numRows/blockDim.y)) + 1, blockDim.y>>>( nn->layers[l]->biases->m, nn->layers[l]->delta->m, nn->alpha,numRows, minibatch_size);        
        cudaDeviceSynchronize();      
    }
}

