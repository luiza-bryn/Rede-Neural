import numpy as np
from neural_network import NeuralNetwork

# Exemplo para teste - XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])

redeneural = NeuralNetwork(hidden_layers=(4,), hidden_activation=['relu'], 
                           loss='bce', 
                           model_type='binary',
                           learning_rate=0.01,
                           max_iter=10000)

redeneural.fit(X, y)
predictions = redeneural.predict(X)
print("Resultado: ")
print(predictions)