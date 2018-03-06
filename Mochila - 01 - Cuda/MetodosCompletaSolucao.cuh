
#pragma once

//#include "objeto.h"
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//completa a solução com os melhores itens segundo seus indices - randomico segundo o tamanho_RCL
__device__ void max_indice(int number_of_itens, bool *solutionParcial, item *size_of_itens, int bin_capacity, int tamanho_RCL, int &peso, int &valor, int seed) {
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);
	//variavel que ira receber um valor randomico
	int idRand = 0;
	//peso que ja foi alocado na mochila
	int capacidade = peso;

	//instanciando a RCL
	int *rcl;
	rcl = (int *)malloc(tamanho_RCL * sizeof(int));
	if (!rcl) {
		printf("Sem memoria disponivel rcl solução inicial!\n");
		//exit(1);
	}

	int i = 0;
	//itens na RCL
	int cont_rcl = 1;

	while (cont_rcl > 0) {

		cont_rcl = 0;

		for (i = 0; i < number_of_itens; i++) {
			if (solutionParcial[i])
				continue;
			if (size_of_itens[i].peso + capacidade <= bin_capacity) {
				rcl[cont_rcl] = i;
				cont_rcl++;
			}
			if (cont_rcl == tamanho_RCL) {
				break;
			}
		}

		if (cont_rcl > 0) {
			idRand = curand(&state) % cont_rcl;
			solutionParcial[rcl[idRand]] = 1;
			capacidade += size_of_itens[rcl[idRand]].peso;
			valor += size_of_itens[rcl[idRand]].valor;
			peso += size_of_itens[rcl[idRand]].peso;
		}

	}
	free(rcl);
}

//completa a solução maximizando o valor segundo o ganho de cada item - coloca sempre o item que maximiza o ganho -  nao aleatorio
__device__ void max_valor(int number_of_itens, bool *solutionParcial, item *size_of_itens, int bin_capacity, int &peso, int &valor) {

	//variavel que ira receber um valor randomico
	int max_valor = valor;
	//peso que ja foi alocado na mochila
	int capacidade = peso;

	int aux_id = 0;

	int i = 0;

	while (aux_id > -1) {
		aux_id = -1;
		max_valor = valor;
		capacidade = peso;
		for (i = 0; i < number_of_itens; i++) {
			if (solutionParcial[i])
				continue;
			if (size_of_itens[i].peso + capacidade <= bin_capacity) {
				if (size_of_itens[i].valor + valor > max_valor) {
					max_valor = size_of_itens[i].valor + valor;
					aux_id = i;
				}

			}
		}
		if (aux_id > -1) {
			solutionParcial[aux_id] = 1;
			valor += size_of_itens[aux_id].valor;
			peso += size_of_itens[aux_id].peso;
		}

	}
}
