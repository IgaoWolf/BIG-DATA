import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Carregar o arquivo ARFF
file_path = r'C:\Users\igora\OneDrive\Documents\Faculdade\BigData\2 Bimestre\supermarket.arff'
data, meta = arff.loadarff(file_path)

# Convertendo os dados para um DataFrame
df = pd.DataFrame(data)

# Decodificar os valores da coluna 'total' se estiverem em bytes
df['total'] = df['total'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Mapear 'high' e 'low' para valores numéricos
df['total'] = df['total'].map({'high': 1, 'low': 0})

# Transformar colunas categóricas em numéricas com LabelEncoder
label_encoders = {}
for column in df.columns:
    if df[column].dtype == object or isinstance(df[column].iloc[0], bytes):
        df[column] = df[column].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Separar características (X) e classe alvo (y)
X = df.drop(columns=['total'])  # Retiramos a coluna 'total' como classe alvo
y = df['total']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando a árvore de decisão
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Visualizando a árvore de decisão
plt.figure(figsize=(20,10))  # Ajuste o tamanho da figura conforme necessário
tree.plot_tree(clf, feature_names=X.columns, class_names=["low", "high"], filled=True)
plt.show()

# Fazendo previsões
y_pred = clf.predict(X_test)

# Avaliando o modelo
from sklearn.metrics import classification_report, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Exibir os resultados
print(f"Acurácia: {accuracy}")
print(f"Relatório de Classificação:\n {report}")
