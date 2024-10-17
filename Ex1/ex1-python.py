import pandas as pd
from scipy.io import arff

# Carregar o arquivo ARFF
file_path = r'C:\Users\igora\OneDrive\Documents\Faculdade\BigData\2 Bimestre\supermarket.arff'
data, meta = arff.loadarff(file_path)

# Convertendo os dados para um DataFrame
df = pd.DataFrame(data)

# Obter o número de atributos e uma breve descrição dos itens listados
num_atributos = df.shape[1]
descricao_atributos = df.dtypes

# Exibir informações ao usuário
print(f"Quantidade de atributos: {num_atributos}")
print(descricao_atributos)
