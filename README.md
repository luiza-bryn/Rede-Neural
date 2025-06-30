 # Projeto Rede Neural - Aprendizado de Máquina

**Integrantes:**
- Luiza Bryn Marangoni Guimarães (21200421)  
- Yuri Rodrigues de Souza (21202346)  
- Filipe Ribeiro Rocha (19203808)

---

## Implementação

A implementação consiste em um modelo de rede neural desenvolvido para tarefas de aprendizado de máquina. O projeto utiliza Python as bibliotecas: `pandas`, `numpy` e `scikit-learn` para o tratamento dos dados e operações matemáticas.

> ### A estrutura do código está organizada da seguinte forma:
- `datasets/`
  - `dog/`
    - `clean_dog_move*.csv` Arquivos CSV contendo os dados de movimentação de cães utilizados para treinamento e teste dos modelos.
    - [`Data_description`](datasets/dog/Data_description) Arquivo de texto contendo a descrição dos dados presentes nos arquivos CSV
    - [`DogInfo.csv`](datasets/dog/DogInfo.csv) Arquivo CSV contendo informações detalhadas sobre cada cão.
  - `pokemon/`
    - [`combats.csv`](datasets/pokemon/combats.csv) Arquivo CSV contendo informações sobre batalhas entre diferentes Pokémon.
    - [`pokemon_alopez237.csv`](datasets/pokemon/pokemon_alopez237.csv) Arquivo CSV contendo informações detalhadas sobre diferentes Pokémon, como atributos, tipos e caracteristicas.
    - [`pokemon_id_each_team.csv`](datasets/pokemon/pokemon_id_each_team.csv) Arquivo CSV que associa cada batalha a um identificador de Pokémon em cada equipe participante.
    - [`pokemon.csv`](datasets/pokemon/pokemon.csv) Arquivo CSV contendo informações detalhadas sobre diferentes Pokémon, como atributos, tipos e caracteristicas.
    [`team_combat.csv`](datasets/pokemon/team_combat.csv) Arquivo CSV que associa o Id de cada pokemon em combate e o vencedor.
  - `validation/` Contém arquivos csv utilizados na validação do rede neural implementada.
    - [`heart.csv`](datasets/validation/heart.csv)
    - [`house_price_regression_dataset`](datasets/validation/house_price_regression_dataset.csv)
    - [`Iris.csv`](datasets/validation/Iris.csv)

- `models/`
  - `binary_class/` 
    - [`model.ipynb`](models/binary_class/model.ipynb) Arquivo Jupyter Notebook contendo o treinamento do modelo para classificação binária.
  - `multi_class/`
    - [`clean_data.ipynb`](models/multi_class/clean_data.ipynb) Arquivo Jupyter Notebook contendo uma redução bruta dos dados
    - [`multiclass.ipynb`](models/multi_class/multiclass.ipynb) Arquivo Jupyter Notebook contendo o treinamento do modelo para classificação multiclasse
  - `regression/`
    - [`model.ipynb`](models/regression/model.ipynb) Arquivo Jupyter Notebook contendo o treinamento do modelo para regressão.
  - `validation/` Contém Jupyter Notebooks que comparam os resultados da nossa implementação com os do scikit-learn em conjuntos de dados reduzidos.
    - [`nn_binary_validation`](models/validation/nn_binary_validation.ipynb)
    - [`nn_multiclass_validation`](models/validation/nn_multiclass_validation.ipynb)
    - [`nn_regrassion_validation`](models/validation/nn_regression_validation.ipynb)

- `neural_network_lib/`
  - [`neural_network.py`](neural_network_lib/neural_network.py) Arquivo Python contendo a implementação da biblioteca de rede neural desenvolvida. Inclui a classe e métodos necessários para a construção e treinamento.
  - [`backpropagation.py`](neural_network_lib/backpropagation.py) Arquivo Python responsável pela implementação do algoritmo de retropropagação.
  - [`activation_function.py`](neural_network_lib/activation_function.py) Arquivo Python que define as funções de ativação utilizadas na rede neural.
  - [`loss.py`](neural_network_lib/loss.py) Arquivo Python responsável pela implementação das funções de cálculo de perda.

---

## Orientações

- Certifique-se de ter Python 3.8+ instalado.
- Crie um ambiente virtual com:
  ```bash
  python -m venv venv
  source venv/bin/activate  # ou venv\Scripts\activate no Windows
  ```
- Instale as dependências:
  ```bash
  pip install -r requirements.txt
  ```

## Instruções de uso

  - Importação do modulo:
  ```python
  from neural_network_lib.neural_network import NeuralNetwork
  ```

  - Criação e treinamento:
  ```python
  model = NeuralNetwork(
      hidden_layers=(4, 4)
      hidden_activation=['relu', 'relu']
      loss='bce'
      model_type='binary'
      learning_rate=0.01
      max_iter=100000
      random_state=42
  )

  model.fit(X_train, y_train)

  pred = model.predict(X_test)
  ```

  ### Parâmetros do construtor
  #### `NeuralNetwork()`
  | Parâmetro         | Tipo            | Exemplo  | Descrição                                                                                      
  |-------------------|-----------------|----------|--------------------------------------------------------------------------------------------------
  | hidden_layers     | Tuple[int, ...] | (4, )    | Tamanho de cada camada oculta                                                                    
  | hidden_activation | List[str]       | ['relu'] | Função de ativação para cada camada oculta.<br> Podendo ser: `sigmoid`, `tahn`, `identity`, `relu` e `softmax` 
  | loss              | str             | 'bce'    | Função de perda. Podendo ser: `mse`, `bce` ou `cce`                                                    
  | model_type        | str             | 'binary' | Tipo do problema. Podendo ser: `binary`, `regression` ou `multiclass`                                  
  | learning_rate     | float           | 0.01     | Taxa de aprendizado                                                                              
  | max_iter          | int             | 100000   | Número maximo de iterações/epocas                                                                
  | random_state      | int             | 42       | seed para reprodutibilidade                                                                      
  
 
  ### Parâmetros dos metodos
  #### `fit(X, y)`
  | Parâmetro | tipo       | Formato                 | Descrição                                                    |
  |-----------|------------|-------------------------|--------------------------------------------------------------|
  | X         | np.ndarray | (n_amostas, n_features) | Array N-dimensional com as amostas                           |
  | y         | np.ndarray | (n_amostras,)           | Array N-dimensional contendo somente os rótulos das amostras |

  #### `predict(X)`
  | Parâmetro | Tipo       | Formato                  | Descrição                                            |
  |-----------|------------|--------------------------|------------------------------------------------------|
  | X         | np.ndarray | (n_amostras, n_features) | Array N-dimensional com as amostras a serem preditas |


  ### Saída
  | Retorno | Tipo       | Descrição                                                           |
  |---------|------------|---------------------------------------------------------------------|
  | output  | np.ndarray | Saída predita pelo modelo. Podendo variar entre o tipo do problema <br> `Classificação binária`: vetor com valores 0 ou 1. <br> `Regressão`: vetor com valores contínuos. <br> `Classificação multiclasse`: vetor com o índice da classe prevista. |