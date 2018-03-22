
__device__ void copia_solucao(int &current_valor, int &current_peso, bool *current_soluction, int num, int peso_parcial, int valor_parcial, bool *soluctionParcial) {
	for (int i = 0; i < num; i++) {
		current_soluction[i] = soluctionParcial[i];
	}
	current_valor = valor_parcial;
	current_peso = peso_parcial;
}

__device__ void SA(int number_of_itens, bool *solutionParcial, int bin_capacity, item *size_of_itens, int seed, int valor_parcial, int peso_parcial, int temperatura, int decaimento_temperatura, int tamanho_RCL, int interacao) {
	curandState_t state;
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);
	//salvando a solucao gerada inicialmente
	int temperaturaFinal = temperatura;

	bool *current_soluctoin;
	current_soluctoin = (bool *)malloc(number_of_itens * sizeof(bool));

	if (!current_soluctoin) {
		printf("Sem memoria disponivel!\n");
		//exit(1);
	}
	int currente_valor;
	int current_peso;


	bool *sa_soluctoin;
	sa_soluctoin = (bool *)malloc(number_of_itens * sizeof(bool));

	if (!sa_soluctoin) {
		printf("Sem memoria disponivel!\n");
		//exit(1);
	}
	int sa_valor;
	int sa_peso;


	copia_solucao(currente_valor, current_peso, current_soluctoin, number_of_itens, peso_parcial, valor_parcial, solutionParcial);
	copia_solucao(sa_valor, sa_peso, sa_soluctoin, number_of_itens, peso_parcial, valor_parcial, solutionParcial);

	int i = 0;
	int delta = 0;

	while (temperaturaFinal > 0) {
		for (i = 0; i < interacao; i++) {
			//gera vizinho (retiro um elemento sorteando como roleta, e coloco o elemento com o melhor indice de ganho disponivel)
			roleta(solutionParcial, size_of_itens, number_of_itens, peso_parcial, valor_parcial, seed);
			max_indice(number_of_itens, solutionParcial, size_of_itens, bin_capacity, 1, peso_parcial, valor_parcial, seed);

			delta = currente_valor - valor_parcial;

			if (delta < 0) {
				copia_solucao(currente_valor, current_peso, current_soluctoin, number_of_itens, peso_parcial, valor_parcial, solutionParcial);
				if (currente_valor > sa_valor) {
					copia_solucao(sa_valor, sa_peso, sa_soluctoin, number_of_itens, current_peso, currente_valor, current_soluctoin);
				}
			}
			else {

				if (curand(&state) / RAND_MAX < expf(-delta / temperatura)) {
					copia_solucao(currente_valor, current_peso, current_soluctoin, number_of_itens, peso_parcial, valor_parcial, solutionParcial);
				}
			}

			//verifica temperatura
			if (curand(&state) % temperatura < temperaturaFinal) {
				valor_parcial = currente_valor;
				peso_parcial = current_peso;

				for (i = 0; i < number_of_itens; i++) {
					solutionParcial[i] = current_soluctoin[i];
				}

			}
			temperaturaFinal -= decaimento_temperatura;

		}
	}
	copia_solucao(valor_parcial, peso_parcial, solutionParcial, number_of_itens, sa_peso, sa_valor, sa_soluctoin);
}