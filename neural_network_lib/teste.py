import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from neural_network import NeuralNetwork
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# 1. Carregar dados
df = pd.read_csv('neural_network_lib/house_price_regression_dataset.csv')  # ex: 'houses.csv'

# 2. Separar variáveis
X = df.drop('House_Price', axis=1).values
y = df['House_Price'].values.reshape(-1, 1)

# 3. Normalizar (escala entre 0 e 1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

nn = NeuralNetwork(
    hidden_layers=(4, 4),
    activation=['relu', 'identity'],
    loss='mse',
    model_type='regression',
    learning_rate=0.1, 
    max_iter=1000000
)

nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
y_pred_nn = scaler_y.inverse_transform(y_pred_nn)
y_test_true = scaler_y.inverse_transform(y_test)


# --- sklearn MLPRegressor ---

mlp = MLPRegressor(hidden_layer_sizes=(4, 4), activation='relu',
                   solver='adam', learning_rate_init=0.001,
                   max_iter=1000000, random_state=42)

mlp.fit(X_train, y_train.ravel())
y_pred_mlp = mlp.predict(X_test)
y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp.reshape(-1, 1))

# Avaliação
print("MSE (Seu modelo):", mean_squared_error(y_test_true, y_pred_nn))
print("R² Score (Seu modelo):", r2_score(y_test_true, y_pred_nn))

print("MSE (MLPRegressor):", mean_squared_error(y_test_true, y_pred_mlp))
print("R² Score (MLPRegressor):", r2_score(y_test_true, y_pred_mlp))