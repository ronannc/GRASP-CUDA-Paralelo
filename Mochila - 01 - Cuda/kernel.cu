/*Bin Packing - two Dimensional
Problema da mochila com duas dimensoes.
Tal problema consiste em escolher o maior numero de itens que comportem em uma mochila (compartimento)

O dataset osado foi o presente na biblioteca ORLIB (OR-Library)

Cada arquivo contem:
Numero de problemas teste
Identificador do Problema
Capacidade do compartimento, numero de itens, numero de itens presentes na melhor solução conhecida.
Tamanho dos itens.
*/

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "objeto.h"
#include "ParallelGRASP.cuh"
#include "EntradaDeDados.h"
#include "Selection_Sort.h"
#include "time.h"

cudaError_t parallel_GRASP(int max_iter, int number_of_itens, int bin_capacity, item *size_of_itens, bool *soluctions, int threads, int blocks, int temperatura, int decaimento_temperatura, int tamanho_RCL, int seed);

int main() {
	//semente para gerador de numeros aleatorios
	srand(time(NULL));
	int seed = rand();
	
	//temperatura inicial e o decaimento da temperatura para SA
	int temperatura = 100;
	int decaimento_temperatura = 1;
	
	//tamanha usado para tornar a geração da solução inicial e busca aleatoria, se 1 fica modo guloso
	int tamanho_RCL = 10;

	//numero de iterações que o GRASP fara
	int max_iter = 10;
	
	//quantidade de threads e blocos
	int threads = 10;
	int blocks = 10;

	// numero de elementos
	int quantidade_itens = 0;

	// capacidade mochia
	int capacidade_mochila = 0;

	/*recebendo entrada, tamanho de cada item*/
	entrada_dados(quantidade_itens, capacidade_mochila);

	/*Vetor usado para guardar peso e valor e se esta ou nao na mochila*/
	item *itens;
	itens = (item *)malloc(quantidade_itens * sizeof(item));

	if (!itens) {
		printf("Sem memoria disponivel! (itens)\n");
		exit(1);
	}

	entrada_dados_vetor(itens, quantidade_itens);

	//ordenando os dados com relação ao valor/peso
	selection_sort(itens, quantidade_itens);

	//for (int i = 0; i < quantidade_itens; i++) {
	//	printf("%d %d\n", itens[i].peso, itens[i].valor);
	//}

	//system("pause");

	printf("====== Bin Packing - Bi Dimensional ======\n");
	printf("\n");
	printf("              IFMG - Formiga            \n");
	printf(" Desenvolvido por: Ronan Nunes Campos   \n");
	printf(" Matricula: 0011919                     \n");
	printf("\n");
	printf(" Dados do problema           \n\n");
	printf(" Numero de Itens: %d                    \n", quantidade_itens);
	printf(" Capacidade Mochila: %d                 \n", capacidade_mochila);
	printf(" Numero de Threads: %d                   \n", threads);
	printf(" Numero de Blocos: %d                    \n", blocks);
	printf(" Numero de Iterações: %d                \n", max_iter);
	printf(" Geradas %d solucões\n", threads * blocks * max_iter);
	printf(" Rodando na GPU                          \n");
	printf("===========================================\n");

	//vetor para guardar o id dos elementos presentes na suloção
	bool *soluctions;
	soluctions = (bool *)malloc(quantidade_itens * threads * blocks * sizeof(bool));

	if (!soluctions) {
		printf("Sem memoria disponivel! (soluctions)\n");
		exit(1);
	}
	//iniciando a solução com todos os itens fora da mochila == 0
	for (int i = 0; i < quantidade_itens * threads * blocks; i++) {
		soluctions[i] = 0;
	}

	// Rodando GRASP em paralelo.
	clock_t t0, tf;
	double tempo_gasto;
	int max_valor = 0; int valor = 0; int cont = 0; int aux_id = 0; int cont_id = 0;
	t0 = clock();
	cudaError_t cudaStatus = parallel_GRASP(max_iter, quantidade_itens, capacidade_mochila, itens, soluctions, threads, blocks, temperatura, decaimento_temperatura, tamanho_RCL, seed);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "parallel_GRASP failed!");
		system("pause");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		system("pause");
		return 1;
	}

	

	for (int i = 0; i < quantidade_itens * threads * blocks; i++) {
		
		if (cont < quantidade_itens) {
			if (soluctions[i] == 1) {
				valor += itens[cont].valor;
			}
			cont++;
		}
		
		if(cont == quantidade_itens){
			
			if (valor > max_valor) {
				max_valor = valor;
				aux_id = cont_id;
			}
			cont_id++;

			cont = 0; valor = 0;
		}
	}
	tf = clock();

	tempo_gasto = ((double)(tf - t0)) / CLOCKS_PER_SEC;
	printf("\n");
	printf("Tempo total gasto: %lf s\n", tempo_gasto);

	printf("===========================================\n\n");
	printf("max valor: %d\n", max_valor);
	for (int i = quantidade_itens * aux_id; i < quantidade_itens * (aux_id + 1); i++) {
		printf("%d ",soluctions[i]);
	}

	printf("\n");
	printf("fim :)\n\n");

	free(itens);
	free(soluctions);
	system("pause");
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t parallel_GRASP(int max_iter, int quantidade_itens, int capacidade_mochila, item *itens, bool *soluctions, int threads, int blocks, int temperatura, int decaimento_temperatura, int tamanho_RCL, int seed) {

	cudaError_t cudaStatus;

	// Get device properties
	int cuda_device = 0;

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, cuda_device);
	int cdpCapable = (properties.major == 3 && properties.minor >= 5) || properties.major >= 4;

	printf("GPU device %s has compute capabilities (SM %d.%d)\n", properties.name, properties.major, properties.minor);

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

	bool *dev_soluctions;
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
	cudaStatus = cudaMemcpy(dev_soluctions, soluctions, quantidade_itens  * blocks * threads * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_soluctions failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	parallelGRASP << <blocks, threads >> >(max_iter, quantidade_itens, capacidade_mochila, dev_itens, dev_soluctions, temperatura, decaimento_temperatura, tamanho_RCL, seed);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaDeviceSynchronize();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//goto Error;
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
