/*Funções para ler e quardar as entradas*/
void entrada_dados(int &number_of_itens, int &bin_capacity) {

	FILE *arq;
	arq = fopen("knapPI_1_1000_1000_1.txt", "rt");
	fscanf(arq, "%d %d\n", &number_of_itens, &bin_capacity);
}

void entrada_dados_vetor(item *itens, int number_of_itens) {

	FILE *arq;
	arq = fopen("knapPI_1_1000_1000_1.txt", "rt");
	int aux = 0;
	int aux1 = 0;
	fscanf(arq, "%d %d\n", &number_of_itens, &aux);
	for (int i = 0; i < number_of_itens; i++) {
		fscanf(arq, "%d %d\n", &aux, &aux1);
		itens[i].peso = aux;
		itens[i].valor = aux1;
	}
}