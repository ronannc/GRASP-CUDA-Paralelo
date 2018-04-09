#pragma once
#include "ParallelGRASP.cuh"

// Helper function for using CUDA to add vectors in parallel.
cudaError_t parallel_GRASP(int max_iter, int quantidade_itens, int capacidade_mochila, item *itens, bool *soluctions, int threads, int blocks, int temperatura, int decaimento_temperatura, int tamanho_RCL, int seed) {

	cudaError_t cudaStatus;

	// Get device properties
	int cuda_device = 0;

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, cuda_device);
	int cdpCapable = (properties.major == 3 && properties.minor >= 5) || properties.major >= 4;

	//printf("GPU device %s has compute capabilities (SM %d.%d)\n", properties.name, properties.major, properties.minor);

	if (!cdpCapable) {
		printf("this app requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...\n");
		system("pause");
		exit(0);
	}

	item *dev_itens;
	cudaStatus = cudaMalloc((void**)&dev_itens, quantidade_itens * sizeof(item));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_size_of_itens failed!");
		goto Error;
	}

	bool* dev_soluctions;
	cudaStatus = cudaMalloc((void**)&dev_soluctions, quantidade_itens * threads * blocks * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_vectorValor failed!");
		goto Error;
	}

	// Copy output vector from host to device memory.
	cudaStatus = cudaMemcpy(dev_itens, itens, quantidade_itens * sizeof(item), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy size_of_itens failed!");
		goto Error;
	}

	// Copy output vector from host to device memory.
	cudaStatus = cudaMemcpy(dev_soluctions, soluctions, quantidade_itens * blocks * threads * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_soluctions failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	parallelGRASP << <blocks, threads >> >(max_iter, quantidade_itens, capacidade_mochila, dev_itens, dev_soluctions, temperatura, decaimento_temperatura, tamanho_RCL, seed);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(soluctions, dev_soluctions, quantidade_itens * blocks * threads * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy soluctions failed!, retornou: %d", cudaStatus);
		goto Error;
	}

Error:
	cudaFree(dev_soluctions);
	cudaFree(dev_itens);

	return cudaStatus;
}
