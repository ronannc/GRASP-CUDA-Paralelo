#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//retira os itens com menor indice de ganho - aleatorio segundo tamanho_RCL
__device__ void retira_menor_indice(int number_of_itens, bool *solutionParcial, item *size_of_itens, int &peso, int &valor, int tamanho_RCL, int seed) {

	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	float peso_valor = (float)valor;
	int idAux = 0;
	int	idRand = 0;

	int *rcl;
	rcl = (int *)malloc(tamanho_RCL * sizeof(int));
	if (!rcl) {
		printf("Sem memoria disponivel rcl SA id!\n");
		//exit(1);
	}

	//pega os n itens com menor relacao peso valor
	for (int i = 0; i < tamanho_RCL; i++) {
		for (int j = 0; j < number_of_itens; j++) {
			if (solutionParcial[j]) {
				if (((float)size_of_itens[j].valor / (float)size_of_itens[j].peso) < peso_valor) {
					idAux = j;
					peso_valor = (float)size_of_itens[j].valor / (float)size_of_itens[j].peso;
				}
			}
		}
		solutionParcial[idAux] = 0;
		rcl[i] = idAux;
		peso_valor = valor;
	}

	idRand = curand(&state) % tamanho_RCL;
	//printf("idRand %d\n", idRand);
	for (int i = 0; i < tamanho_RCL; i++) {
		solutionParcial[rcl[i]] = 1;
	}

	//retiro o elemento da solucao
	solutionParcial[rcl[idRand]] = 0;
	valor -= size_of_itens[rcl[idRand]].valor;
	peso -= size_of_itens[rcl[idRand]].peso;

	//printf("depois de tirar o numero\n");
	//printf("numero retirado peso %d valor %d\n", size_of_itens[rcl[idRand]].peso, size_of_itens[rcl[idRand]].valor);
	//revisao(number_of_itens, solutionParcial, size_of_itens);

	free(rcl);
}

//retira o item com maior peso - aleatorio segundo tamanho_RCL
__device__ void retira_maior_peso(int number_of_itens, bool *solutionParcial, item *size_of_itens, int &peso, int &valor, int tamanho_RCL, int seed) {

	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	int peso_valor = peso;
	int idAux = 0;
	int idRand = 0;

	int *rcl;
	rcl = (int *)malloc(tamanho_RCL * sizeof(int));
	if (!rcl) {
		printf("Sem memoria disponivel rcl SA id!\n");
		//exit(1);
	}

	//pega os n itens com maior peso
	for (int i = 0; i < tamanho_RCL; i++) {
		for (int j = 0; j < number_of_itens; j++) {
			if (solutionParcial[j]) {
				if (size_of_itens[j].peso < peso_valor) {
					idAux = j;
					peso_valor = size_of_itens[j].peso;
				}
			}
		}
		solutionParcial[idAux] = 0;
		rcl[i] = idAux;
		peso_valor = peso;
	}

	idRand = curand(&state) % tamanho_RCL;
	//printf("idRand %d\n", idRand);
	for (int i = 0; i < tamanho_RCL; i++) {
		solutionParcial[rcl[i]] = 1;
	}

	//retiro o elemento da solucao
	solutionParcial[rcl[idRand]] = 0;
	valor -= size_of_itens[rcl[idRand]].valor;
	peso -= size_of_itens[rcl[idRand]].peso;

	//printf("depois de tirar o numero\n");
	//printf("numero retirado peso %d valor %d\n", size_of_itens[rcl[idRand]].peso, size_of_itens[rcl[idRand]].valor);
	//revisao(number_of_itens, solutionParcial, size_of_itens);

	free(rcl);
}

//retira o item com menor valor - aleatorio segundo tamanho_RCL
__device__ void retira_menor_valor(int number_of_itens, bool *solutionParcial, item *size_of_itens, int &peso, int &valor, int tamanho_RCL, int seed) {

	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	int peso_valor = valor;
	int idAux = 0;
	int idRand = 0;

	int *rcl;
	rcl = (int *)malloc(tamanho_RCL * sizeof(int));
	if (!rcl) {
		printf("Sem memoria disponivel rcl SA id!\n");
		//exit(1);
	}

	//pega os n itens com maior peso
	for (int i = 0; i < tamanho_RCL; i++) {
		for (int j = 0; j < number_of_itens; j++) {
			if (solutionParcial[j]) {
				if (size_of_itens[j].valor < peso_valor) {
					idAux = j;
					peso_valor = size_of_itens[j].valor;
				}
			}
		}
		solutionParcial[idAux] = 0;
		rcl[i] = idAux;
		peso_valor = valor;
	}

	idRand = curand(&state) % tamanho_RCL;
	//printf("idRand %d\n", idRand);
	for (int i = 0; i < tamanho_RCL; i++) {
		solutionParcial[rcl[i]] = 1;
	}

	//retiro o elemento da solucao
	solutionParcial[rcl[idRand]] = 0;
	valor -= size_of_itens[rcl[idRand]].valor;
	peso -= size_of_itens[rcl[idRand]].peso;

	//printf("depois de tirar o numero\n");
	//printf("numero retirado peso %d valor %d\n", size_of_itens[rcl[idRand]].peso, size_of_itens[rcl[idRand]].valor);
	//revisao(number_of_itens, solutionParcial, size_of_itens);

	free(rcl);
}\

__device__ void roleta(bool *solutionParcial, item *size_of_itens, int number_of_itens, int &peso, int &valor, int seed) {
	struct roleta {
		int id;
		float ganho;
	};

	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	float soma_ganhos = 0;
	int tamanho = 0;

	for (int i = 0; i < number_of_itens; i++) {
		if (solutionParcial[i]) {
			++tamanho;
			soma_ganhos += size_of_itens[i].ganho;
		}

	}

	roleta *iten_roleta;
	iten_roleta = (roleta *)malloc(tamanho * sizeof(roleta));

	roleta aux;
	aux.ganho = 0;
	aux.id = 0;
	float anterior = 0;

	for (int j = 0; j < tamanho; j++) {
		for (int i = 0; i < number_of_itens; i++) {
			if (solutionParcial[i]) {
				if (size_of_itens[i].ganho > aux.ganho) {
					aux.ganho = size_of_itens[i].ganho;
					aux.id = i;
				}
			}
		}
		iten_roleta[j].ganho = (soma_ganhos - aux.ganho) + anterior;
		iten_roleta[j].id = aux.id;
		anterior += soma_ganhos - aux.ganho;
	}

	float id_rand = curand(&state) % (int)anterior;
	int aux_id = 0;

	for (int i = 0; i < tamanho; i++) {
		if (iten_roleta[i].ganho >= id_rand) {
			aux_id = iten_roleta[i].id;
			break;
		}
	}

	solutionParcial[aux_id] = 0;
	valor -= size_of_itens[aux_id].valor;
	peso -= size_of_itens[aux_id].peso;
}
