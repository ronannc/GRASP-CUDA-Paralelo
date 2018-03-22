#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "MetodosCompletaSolucao.cuh"
#include "MetodosRetiraItem.cuh"
#include "Simulated_Annealing.cuh"
#include "GRASPLib.cuh"


//gera a solução inicial - aleatorizado
__device__ void GreedyRandomizedConstruction(int number_of_itens, bool *solutionParcial, item *size_of_itens, int bin_capacity, int seed, int tamanho_RCL, int &peso, int &valor);

//faz a busca local
__device__ void LocalSearch(int number_of_itens, bool *solutionParcial, int bin_capacity, item *size_of_itens, int seed, int &valor_parcial, int &peso_parcial, int tamanho_RCL);

//Atualiza a solução final com a melhor solução maximizando o ganho (valor)
__device__ void UpdateSolution(bool *solutionParcial, bool *solutionFinal, int number_of_itens);

//GRASP 
__global__ void parallelGRASP(int max_iter, int number_of_itens, int bin_capacity, item *size_of_itens, bool *soluctions, int temperatura, int decaimento_temperatura, int tamanho_RCL, int seed){

	/*id da thread corrente*/
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//solução parcial da iteração/thread
	bool *solutionParcial;
	solutionParcial = (bool *)malloc(number_of_itens * sizeof(bool));
	//verificando se foi alocado
	if (!solutionParcial) {
		printf("Sem memoria disponivel id: %d!\n", idx);
		//exit(1);
	}
	//inicializando - todos itens fora da solução == 0
	for (int i = 0; i < number_of_itens; i++) {
		solutionParcial[i] = 0;
	}

	int  i, j = 0;

	int valor_parcial = 0, peso_parcial = 0, max_valor = 0;

	for (i = 0; i < max_iter; i++) {

		/*inicio grasp*/

		//gera solução inicial
		GreedyRandomizedConstruction(number_of_itens, solutionParcial, size_of_itens, bin_capacity, seed + i + idx, tamanho_RCL, peso_parcial, valor_parcial);

		//printf("Solucao gerada inicialmente\n");
		//printf("peso: %d  valor: %d\n\n", peso_parcial, valor_parcial);

		//revisao(number_of_itens, solutionParcial, size_of_itens);

		//faz a busca local tentando melhorar a solução gerada
		LocalSearch(number_of_itens, solutionParcial, bin_capacity, size_of_itens, seed + i + idx, valor_parcial, peso_parcial, tamanho_RCL);
		// SA(number_of_itens, solutionParcial, bin_capacity, size_of_itens, seed + i + idx, valor_parcial, peso_parcial, temperatura, decaimento_temperatura, tamanho_RCL, seed);
		
		//printf("Solucao gerado depois da busca \n");
		//printf("peso: %d  valor: %d\n\n", peso_parcial, valor_parcial);

		//revisao(number_of_itens, solutionParcial, size_of_itens);

		//atualiza a solução
		if (valor_parcial > max_valor) {
			max_valor = valor_parcial;
			UpdateSolution(solutionParcial, soluctions, number_of_itens);
		}

		//a cada iteração reseta peso, valor e a solução parcial
		peso_parcial = 0;
		valor_parcial = 0;

		for (j = 0; j < number_of_itens; j++) {
			solutionParcial[j] = 0;
		}
	}

	free(solutionParcial);
}

__device__ void GreedyRandomizedConstruction(int number_of_itens, bool *solutionParcial, item *size_of_itens, int bin_capacity, int seed, int tamanho_RCL, int &peso, int &valor){

	//completa com os melhor elementos segundo seus indeces
	max_indice(number_of_itens, solutionParcial, size_of_itens, bin_capacity, tamanho_RCL, peso, valor, seed);
}

__device__ void LocalSearch(int number_of_itens, bool *solutionParcial, int bin_capacity, item *size_of_itens, int seed, int &valor_parcial, int &peso_parcial, int tamanho_RCL){

	int valor_parcial_busca = valor_parcial;

	bool flag = true;

	while (flag) {
		flag = false;

		roleta(solutionParcial, size_of_itens, number_of_itens, peso_parcial, valor_parcial, seed);

		//retira_menor_indice(number_of_itens, solutionParcial, size_of_itens, peso_parcial, valor_parcial, 5, seed);

		//retira_maior_peso(number_of_itens, solutionParcial, size_of_itens, peso_parcial, valor_parcial, 1, seed);

		//retira_menor_valor(number_of_itens, solutionParcial, size_of_itens, peso_parcial, valor_parcial, 1, seed);

		//max_valor(number_of_itens, solutionParcial, size_of_itens, bin_capacity, peso_parcial, valor_parcial);

		max_indice(number_of_itens, solutionParcial, size_of_itens, bin_capacity, 1, peso_parcial, valor_parcial, seed);

		if (valor_parcial > valor_parcial_busca) {
			valor_parcial_busca = valor_parcial;
			flag = true;
		}
	}
	
}

__device__ void UpdateSolution(bool *solutionParcial, bool *solutionFinal, int number_of_itens){

	/*id da thread corrente*/
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int j = number_of_itens * idx; j < number_of_itens * (idx + 1); j++) {
		if (solutionParcial[j - (number_of_itens * idx)]){
			solutionFinal[j] = 1;
		}
		else {
			solutionFinal[j] = 0;
		}
	}
}