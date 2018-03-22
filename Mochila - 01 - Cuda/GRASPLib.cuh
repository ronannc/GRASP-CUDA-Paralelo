#pragma once

#include <stdio.h>
#include <stdlib.h>

__device__ void revisao(int number_of_itens, bool *solutionParcial, item *size_of_itens) {
	int peso_parcial = 0;
	int valor_parcial = 0;

	for (int j = 0; j < number_of_itens; j++) {
		if (solutionParcial[j]) {
			peso_parcial += size_of_itens[j].peso;
			valor_parcial += size_of_itens[j].valor;
		}
	}

	printf("(revisao)\n");
	printf("peso: %d  valor: %d\n\n", peso_parcial, valor_parcial);
}