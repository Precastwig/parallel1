#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 1024
//#define COMMENTS

__global__ void d_winAverage(int *A, float *B, size_t size, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		B[i] = ((float)A[i] - (float)A[max(0, i - n)]) / (float)n;
	}
}

__host__ void winAverage(int *A, float *B, size_t size, int n) {
	for (int i = 0; i < size; i++) {
		B[i] = ((float)A[i] - (float)A[max(0, i - n)]) / (float)n;
	}
}

__global__ void d_scan(int *A, int *B, size_t size) {
	//First copy into shared memory
	__shared__ float temp[BLOCK_SIZE];

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		temp[threadIdx.x] = A[i];
	}
	//Now reduction
	for (uint stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < BLOCK_SIZE) {
			temp[index] += temp[index - stride];
		}
	}
	//Now distribution
	for (uint stride = BLOCK_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < BLOCK_SIZE) {
			temp[index + stride] += temp[index];
		}
	}
	//And copy into the output0078
	__syncthreads();
	if (i < size) {
		B[i] = temp[threadIdx.x];
	}
}

__global__ void d_extract(int *B, int *Sum, size_t size) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if ((i + 1) * BLOCK_SIZE < size) {
		Sum[i] = B[((i + 1) * BLOCK_SIZE) - 1];
	}
}

__global__ void d_add(int *Sum_scanned, int *Y, size_t size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (blockIdx.x > 0 && i < size) {
		Y[i] += Sum_scanned[blockIdx.x - 1];
	}

}

__host__ void scan(int *A, int *B, size_t size) {
	for (int i = 0; i < size; i++) {
		B[i] = B[i - 1] + A[i];
	}
}

int main(void) {
	cudaError_t err = cudaSuccess;
	//DEFINING SOME VARIABLES TO EDIT AND STUFF
	int vectorSizes = 1000000;
	//uint MAX_ARRAY_SIZE = 4 * BLOCK_SIZE * BLOCK_SIZE;
	//printf("%d", MAX_ARRAY_SIZE);
	int windowsize = 1000;
	size_t size = vectorSizes * sizeof(int);
	size_t smallsize = (vectorSizes / BLOCK_SIZE) * sizeof(int);
	size_t verysmallsize = (vectorSizes / (BLOCK_SIZE * BLOCK_SIZE))
			* sizeof(int);
	size_t floatsize = vectorSizes * sizeof(float);
	//Allocate host vector
	int *h_A = (int *) malloc(size);
	if (h_A == NULL) {
		fprintf(stderr, "Failed to allocate host vector\n");
		exit(EXIT_FAILURE);
	}

	//Initialise the vector
	for (int i = 0; i < vectorSizes; i++) {
		//time_t t;
		//srand((unsigned) time(&t));
		//h_A[i] = rand() % 10;

		h_A[i] = 9;
	}

	//Allocate the device vectors
	int *d_A = NULL;
	err = cudaMalloc((void **) &d_A, size);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device vector");
		exit(EXIT_FAILURE);
	}

	int *d_B = NULL;
	err = cudaMalloc((void **) &d_B, size);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device vector");
		exit(EXIT_FAILURE);
	}

	int *d_Sum1 = NULL;
	err = cudaMalloc((void **) &d_Sum1, smallsize);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated Sum1");
		exit(EXIT_FAILURE);
	}

	int *d_Sum1_scanned = NULL;
	err = cudaMalloc((void **) &d_Sum1_scanned, smallsize);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated Sum1_scanned");
		exit(EXIT_FAILURE);
	}

	int *d_Sum2 = NULL;
	err = cudaMalloc((void **) &d_Sum2, verysmallsize);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated Sum2");
		exit(EXIT_FAILURE);
	}

	int *d_Sum2_scanned = NULL;
	err = cudaMalloc((void **) &d_Sum2_scanned, verysmallsize);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated Sum2_scanned");
		exit(EXIT_FAILURE);
	}

	float *d_C = NULL;
	err = cudaMalloc((void **) &d_C, floatsize);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device vector");
		exit(EXIT_FAILURE);
	}

	//Copy the host input vector A into device vector A

	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy host data to device data");
		exit(EXIT_FAILURE);
	}

	int *h_B = (int *) malloc(size);
	float *h_C = (float *) malloc(floatsize);

	int blocksPerGrid = 1 + ((vectorSizes - 1) / BLOCK_SIZE);

	cudaEvent_t start1, stop1, start2, stop2;
	float time1, time2;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	//Do serial version
	cudaEventRecord(start1, 0);
	scan(h_A, h_B, vectorSizes);
	winAverage(h_B,h_C,vectorSizes, windowsize);
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);

	cudaEventElapsedTime(&time1, start1, stop1);
	printf("Serial version: %.4f ms\n", time1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);

	//Do parallel version
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
	d_scan<<<blocksPerGrid, BLOCK_SIZE>>>(d_A, d_B, vectorSizes);
	d_extract<<<blocksPerGrid, BLOCK_SIZE>>>(d_B, d_Sum1, vectorSizes);
	d_scan<<<blocksPerGrid, BLOCK_SIZE>>>(d_Sum1, d_Sum1_scanned, (vectorSizes / BLOCK_SIZE));
	/*d_extract<<<blocksPerGrid, BLOCK_SIZE>>>(d_Sum1_scanned, d_Sum2, vectorSizes / (BLOCK_SIZE * BLOCK_SIZE));
	int *h_Temp2 = (int *) malloc(verysmallsize);
		err = cudaMemcpy(h_Temp2, d_Sum2, verysmallsize, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy host data to device data, %s\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		for(int i = 0; i < vectorSizes / (BLOCK_SIZE * BLOCK_SIZE); i++) {

				printf(
						"Error returned vectors not matching #%d, reg:%d, device:%d\n",
						i, h_A[i], h_Temp2[i]);
		}
	d_scan<<<blocksPerGrid, BLOCK_SIZE>>>(d_Sum2, d_Sum2_scanned, vectorSizes / (BLOCK_SIZE * BLOCK_SIZE));
	d_add<<<blocksPerGrid, BLOCK_SIZE>>>(d_Sum2_scanned, d_Sum1_scanned, vectorSizes / BLOCK_SIZE); */
	d_add<<<blocksPerGrid, BLOCK_SIZE>>>(d_Sum1_scanned, d_B, vectorSizes);
	d_winAverage<<<blocksPerGrid, BLOCK_SIZE>>>(d_B, d_C, vectorSizes,windowsize);
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);

	cudaEventElapsedTime(&time2, start2, stop2);
	printf("Parallel version: %.4f ms\n", time2);

	//Free and destroy everything
	//Check that the result vectors are correct#

	printf("Testing the result vectors\n");
	float *h_Temp = (float *) malloc(floatsize);
	err = cudaMemcpy(h_Temp, d_C, floatsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy host data to device data, %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	bool cont = true;
	int i = 0;
	while (i < vectorSizes && cont == true) {
		if ((int) h_C[i] != (int) h_Temp[i]) {
			printf(
					"Error returned vectors not matching #%d, reg:%d, host:%.4f vs device:%.4f\n",
					i, h_A[i], h_C[i], h_Temp[i]);
			cont = false;
		}
		i++;
	}
	if (cont == true) {
		printf("Both return vectors match!\n");
	}

	free(h_A);
	free(h_B);
	free(h_C);
	free(h_Temp);
	//free(h_Temp2);
#ifdef COMMENTS
	free(h_Temp3);
#endif
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_Sum1);
	cudaFree(d_Sum1_scanned);
	cudaFree(d_Sum2);
	cudaFree(d_Sum2_scanned);
}

