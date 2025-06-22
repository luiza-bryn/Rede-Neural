import numpy as np

# Funcao Logistica
def logistic(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

def logistic_derivative(x):
    sigmoid_derivative = logistic(x) *(1 - logistic(x))
    return sigmoid_derivative

# Funcao Tangente Hiperbolica
def tanh(x):
    hyperbolic = np.tanh(x)
    return hyperbolic

def tanh_derivative(x):
    hyperbolic_derivative = 1 - tanh(x) ** 2
    return hyperbolic_derivative

# Funcao Identidade
def identity(x):
    identity_function = x
    return identity_function

def identity_derivative(x):
    identity_function_derivaitive = 1
    return identity_function_derivaitive

# Unidade Linear Retificada (ReLU)
def relu(x):
    relu_function = np.maximum(x, 0)
    return relu_function

def relu_derivative(x):
    if x >= 0:
        relu_function_derivative = 1
    if x <0:
        relu_function_derivative = 0
    return relu_function_derivative

# Funcao softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_function = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_function

# fonte: https://aimatters.wordpress.com/2020/06/14/derivative-of-softmax-layer/
def softmax_derivative(x):
    x_vector = x.reshape(x.shape[0], 1)
    x_matrix = np.tile(x_vector, x.shape[0])
    x_dir = np.diag(x) - np.dot(x_vector, x_vector.T) #(x_matrix * np.transpose(x_matrix))
    return x_dir

# caso utilize a softmax com o categorical cross entropy o gradiente fica: gradiente = y_pred - y_true

activation_funcions = {
    'sigmoid': (logistic, logistic_derivative),
    'tanh': (tanh, tanh_derivative),
    'identity': (identity, identity_derivative),
    'relu': (relu, relu_derivative),
    'softmax': (softmax, softmax_derivative)
}