void selection_sort(item *num, int tam){

	int i, j, min, peso, valor = 0;
	
	for (i = 0; i < (tam - 1); i++) {

		min = i;
		for (j = (i + 1); j < tam; j++) {

			if (((float)num[j].valor / (float)num[j].peso) > ((float)num[min].valor / (float)num[min].peso)) {
				min = j;
			}
		}
		if (i != min) {

			peso = num[i].peso;
			valor = num[i].valor;
			num[i].peso = num[min].peso;
			num[i].valor = num[min].valor;
			num[min].peso = peso;
			num[min].valor = valor;
		}
	}
}
