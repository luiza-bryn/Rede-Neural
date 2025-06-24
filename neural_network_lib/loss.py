import numpy as np

# Erro Quadratico Medio (MSE)
def  mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

def mean_squared_error_derivative(y_true, y_pred):
    num_samples = y_true.shape[0]
    mse_drivative = -2*(y_true - y_pred) /  num_samples
    return mse_drivative

# Entropia Cruzada Binaria
def binary_cross_entropy(y_true, y_pred):
    # Evitar o resultado de log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def binary_cross_entropy_derivative(y_true, y_pred):
    # Evitar o resultado de log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce_derivative = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
    return bce_derivative

# Entropia Cruzada Categorica
def categorical_cross_entropy(y_true, y_pred):
    # Evitar o resultado de log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return cce

def categorical_cross_entropy_derivative(y_true, y_pred):
    # Evitar o resultado de log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    cce_derivative = -(y_true / y_pred) / y_true.shape[0]
    return cce_derivative

# Dicionario de mapeamento dos nomes
loss_functions = {
    'mse': (mean_squared_error, mean_squared_error_derivative),
    'binary_crossentropy': (binary_cross_entropy,binary_cross_entropy_derivative),
    'categorical_crossentropy': (categorical_cross_entropy, categorical_cross_entropy_derivative)
}