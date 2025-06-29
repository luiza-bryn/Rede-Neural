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
  - `neural_network.py` Arquivo Python contendo a implementação da biblioteca de rede neural desenvolvida. Inclui a classe e métodos necessários para a construção e treinamento.
  - `backpropagation.py` Arquivo Python responsável pela implementação do algoritmo de retropropagação.
  - `activation_function.py` Arquivo Python que define as funções de ativação utilizadas na rede neural.
  - `loss.py` Arquivo Python responsável pela implementação das funções de cálculo de perda.

---

## Orientações

- Certifique-se de ter Python 3.8+ instalado.
- Crie um ambiente virtual com:
  ```bash
  python -m venv venv
  source venv/bin/activate  # ou venv\Scripts\activate no Windows

