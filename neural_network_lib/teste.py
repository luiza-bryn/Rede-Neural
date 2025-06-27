import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. Carregar dados
df = pd.read_csv('C:/Users/liper/Dropbox/1 - UFSC - Sistemas da Informação/2025-01/INE5664 - Aprendizado de Máquina/Rede-Neural/neural_network_lib/Iris.csv')

# 2. Separar variáveis
X = df.drop('label', axis=1).values
y = df['label'].values

# 3. Normalizar (escala entre 0 e 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = LabelEncoder().fit_transform(y)

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

nn = NeuralNetwork(hidden_layers=(16,12), hidden_activation=['relu', 'relu'],
                   loss='cce',
                   model_type='multiclass',
                   learning_rate=0.01,
                   max_iter=10000)

nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)

# --- sklearn MLPRegressor ---

mlp = MLPClassifier(hidden_layer_sizes=(16, 12), activation='relu',
                    learning_rate=0.01,
                    max_iter=10000)

mlp.fit(X_train, y_train.ravel())
y_pred_mlp = mlp.predict(X_test)


# Avaliação do modelo NeuralNetwork customizado
print("Avaliação do modelo NeuralNetwork customizado:")
print("Acurácia:", accuracy_score(y_test, y_pred_nn))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_nn))
print("Relatório de classificação:\n", classification_report(y_test, y_pred_nn))

# Avaliação do modelo MLPClassifier do scikit-learn
print("Avaliação do modelo MLPClassifier do scikit-learn:")
print("Acurácia:", accuracy_score(y_test, y_pred_mlp))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_mlp))
print("Relatório de classificação:\n", classification_report(y_test, y_pred_mlp))