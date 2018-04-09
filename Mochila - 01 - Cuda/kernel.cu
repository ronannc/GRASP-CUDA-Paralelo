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

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
#include "objeto.h"
#include "EntradaDeDados.h"
#include "Selection_Sort.h"
#include "HelperCUDA.h"
#include "SaidaArquivo.h"

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

	/*Vetor usado para guardar peso, valor e ganho (valor/peso)*/
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
	saida_header();

	//for n vezes para executar testes
	for (int k = 0; k < 100; k++) {
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

			if (cont == quantidade_itens) {

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
		saida_body(max_valor, tempo_gasto, max_iter, threads, blocks);
		printf("\n");
		printf("Tempo total gasto: %lf s\n", tempo_gasto);

		printf("===========================================\n\n");
		printf("max valor: %d\n", max_valor);
		for (int i = quantidade_itens * aux_id; i < quantidade_itens * (aux_id + 1); i++) {
			printf("%d ", soluctions[i]);
		}

		printf("\n");
		printf("fim :)\n\n");
	}
	

	free(itens);
	free(soluctions);
	system("pause");
	return 0;
}