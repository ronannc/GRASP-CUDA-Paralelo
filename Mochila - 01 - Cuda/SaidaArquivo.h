#include <stdio.h>

void saida_header() {
	FILE *arq;

	arq = fopen("saida.txt", "a");

	if (arq == NULL)
		printf("Erro, nao foi possivel abrir o arquivo\n");

	else {

		fprintf(arq, "\n\nSaida de Dados - GRASP Problema da Mochila 0 1 em CUDA\nPara instacia: knapPI_1_2000_1000_1.txt\n\n");
	}
	fclose(arq);
}


void saida_body(int valor, double tempo, int maxIter, int threads, int blocks){

	FILE *arq;

	arq = fopen("saida.txt", "a");

	if (arq == NULL)
		printf("Erro, nao foi possivel abrir o arquivo\n");

	else {
		
		fprintf(arq, "Max Iter: %d   Threads: %d   Blocks: %d   Tempo: %.2f   Valor: %d\n", maxIter, threads, blocks, tempo, valor);
	}
	fclose(arq);
}