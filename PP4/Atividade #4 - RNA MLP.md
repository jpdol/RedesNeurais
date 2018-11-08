
# Projeto Prático #4 
## Multilayer Perceptron + GridSearchCV + WheatSeedsDataset


Este Projeto Prático tem o objetivo de conduzir um processo de Aprendizado de Máquina com a tarefa de Classificação Multiclasse que utilize Redes Neurais Artificiais do tipo Multilayer Perceptron para solucionar o problema de classificação de três variedades de trigo (Kama, Rosa, Canadian) a partir dos seguintes dados:
    
    
    Área, Perímetro, Compactude, Comprimento, Largura, Coeficiente de Assimetria e Comprimento do Sulco da Semente
    
estes, encontrados no [WheatSeedsDataset](https://archive.ics.uci.edu/ml/datasets/seeds#).

Com intuito de otimização na busca por melhores parâmetro e hiperparâmetros da RNA, neste projeto, será utilizada uma Busca em Grade que irá variar a função de ativação e número de neurônios nas camadas ocultas

Para a avaliação das RNAs encontradas, a Busca em Grade considerará uma Validação Cruzada com k=3 folds e a acurácia como métrica de desempenho.

Alunos: 
    - Jean Phelipe de Oliveira Lima - 1615080096
    - Rodrigo Gomes de Souza - 1715310022

## Bibliotecas


```python
import pandas as pd
import numpy as np
from math import ceil
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

## Leitura do WhatSeedsDataset


```python
dataset = pd.read_csv('WheatSeedDataset.csv', sep='\t')
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Area</th>
      <th>Perimeter</th>
      <th>Compactness</th>
      <th>Length of Kernel</th>
      <th>Width of Kernel</th>
      <th>Asymmetry Coefficient</th>
      <th>Length of Kernel Groove</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.26</td>
      <td>14.84</td>
      <td>0.8710</td>
      <td>5.763</td>
      <td>3.312</td>
      <td>2.221</td>
      <td>5.220</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.88</td>
      <td>14.57</td>
      <td>0.8811</td>
      <td>5.554</td>
      <td>3.333</td>
      <td>1.018</td>
      <td>4.956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.29</td>
      <td>14.09</td>
      <td>0.9050</td>
      <td>5.291</td>
      <td>3.337</td>
      <td>2.699</td>
      <td>4.825</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.84</td>
      <td>13.94</td>
      <td>0.8955</td>
      <td>5.324</td>
      <td>3.379</td>
      <td>2.259</td>
      <td>4.805</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.14</td>
      <td>14.99</td>
      <td>0.9034</td>
      <td>5.658</td>
      <td>3.562</td>
      <td>1.355</td>
      <td>5.175</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Regra da Pirâmide Geométrica

Implementação da Regra da Pirâmide Geométrica para determinação da quantidade de Neurônios Ocultos

        Nh = α·√(Ni·No) ; Nh = Número de Neurônios Ocultos
                          Ni = Número de Neurônios de Entrada
                          No = Número de Neurônios de Saída
                          α  = Constante (Para o problema em questão, serão adotados α = [0.5, 2, 3])


```python
def piramide_geometrica(ni, no, alfa):
    nh = alfa*((ni*no)**(1/2))
    return ceil(nh)
```

##  Distribuição dos Neurônios em duas Camadas Ocultas

Função para gerar todas as possíveis 2-tuplas que representam o número de neurônios distribuídos por duas camadas ocultas de uma RNA do tipo MLP, dado o número de neurônios ocultos obtidos previamente pela Regra da Pirâmide Geométrica.


```python
def hidden_layers(layers, nh):
    for i in range(1, nh):
        neurons_layers = (i, nh-i)
        layers.append(neurons_layers)
    return layers
```

### Criação de Lista de Camadas Ocultas a Partir da Regra da Pirâmide Geométrica


```python
num_in = 7
num_out = 3
alpha = [0.5, 2, 3]
layers = []
```


```python
for i in range(len(alpha)):
    nh = piramide_geometrica(num_in, num_out, alpha[i])
    print('Para α = %.1f, Nh = %d'%(alpha[i],nh))
    hidden_layers(layers, nh)#insere cada possibilidade de camadas ocultas, dado o numero de neurônios, na lista 'layers'
    
print()
print('Distribuições de Camadas Ocultas:\n')
for i in layers:
    print(i)
```

    Para α = 0.5, Nh = 3
    Para α = 2.0, Nh = 10
    Para α = 3.0, Nh = 14
    
    Distribuições de Camadas Ocultas:
    
    (1, 2)
    (2, 1)
    (1, 9)
    (2, 8)
    (3, 7)
    (4, 6)
    (5, 5)
    (6, 4)
    (7, 3)
    (8, 2)
    (9, 1)
    (1, 13)
    (2, 12)
    (3, 11)
    (4, 10)
    (5, 9)
    (6, 8)
    (7, 7)
    (8, 6)
    (9, 5)
    (10, 4)
    (11, 3)
    (12, 2)
    (13, 1)


## Busca em Grade

São definidos:
    - Parâmetros que devem variar na busca em grade;
    - Número de Folds para validação cruzada;
    - Métrica de desempenho a ser considerada;
    
    
Além disso, é definido o método de otimização a ser utilizado: ***solver = lbfgs***, uma vez que dataset utilizado contém poucos exemplos para cada classe.


```python
parameters = {'solver': ['lbfgs'], 
              'activation': ['identity', 'logistic', 'tanh', 'relu'],
              'hidden_layer_sizes': layers,
              'max_iter':[1000],
              'learning_rate': ['adaptive', 'constant']}

gs = GridSearchCV(MLPClassifier(), 
                  parameters, 
                  cv=3, 
                  scoring='accuracy')
```


```python
x = dataset.drop(['Type'], axis = 1) #Atributos preditores
y = dataset.Type #Atributo Alvo
```

### Treinamento 

Treinamento de todas as combinações de RNAs definidas no GridSearchCV()


```python
gs.fit(x, y)
```




    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=None, shuffle=True, solver='adam', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'solver': ['lbfgs'], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [(1, 2), (2, 1), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1), (1, 13), (2, 12), (3, 11), (4, 10), (5, 9), (6, 8), (7, 7), (8, 6), (9, 5), (10, 4), (11, 3), (12, 2), (13, 1)], 'max_iter': [1000], 'learning_rate': ['adaptive', 'constant']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)



# Resultados

### Acurácia e Parâmetros do melhor modelo:


```python
#Acurácia para o conjunto de testes
print('Acurácia média para os 3 splits de teste:',gs.best_score_)

print('\nParâmetros:')
for key in gs.best_params_.keys():
    print('\t',key, ': ', gs.best_params_[key])
```

    Acurácia média para os 3 splits de teste: 0.9333333333333333
    
    Parâmetros:
    	 activation :  identity
    	 hidden_layer_sizes :  (5, 5)
    	 learning_rate :  adaptive
    	 max_iter :  1000
    	 solver :  lbfgs


### Dataframe - Desempenho de cada RNA


```python
results = pd.DataFrame(gs.cv_results_)
analysis_dict = {}

analysis_dict['hidden_layer_sizes'] = results['param_hidden_layer_sizes']
analysis_dict['activation'] = results['param_activation']
analysis_dict['learning_rate'] = results['param_learning_rate']
analysis_dict['mean_test_accuracy'] = results['mean_test_score']

analysis_dataset = pd.DataFrame(analysis_dict)
analysis_dataset.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hidden_layer_sizes</th>
      <th>activation</th>
      <th>learning_rate</th>
      <th>mean_test_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(1, 2)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(1, 2)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.852381</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(2, 1)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.842857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(2, 1)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.847619</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(1, 9)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.852381</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(1, 9)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.861905</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(2, 8)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(2, 8)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.909524</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(3, 7)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.923810</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(3, 7)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.900000</td>
    </tr>
  </tbody>
</table>
</div>



### Número de RNAs treinadas:


```python
print("Total de RNAs:",len(results))
```

    Total de RNAs: 192


### Top 10 - Melhores RNAs

Melhores RNAs para o problema, ordenadas pela acurácia.


```python
top10 = analysis_dataset.sort_values('mean_test_accuracy', ascending=False)
top10.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hidden_layer_sizes</th>
      <th>activation</th>
      <th>learning_rate</th>
      <th>mean_test_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87</th>
      <td>(9, 5)</td>
      <td>logistic</td>
      <td>constant</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>67</th>
      <td>(8, 2)</td>
      <td>logistic</td>
      <td>constant</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(5, 5)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(7, 3)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>27</th>
      <td>(3, 11)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>31</th>
      <td>(5, 9)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(5, 5)</td>
      <td>identity</td>
      <td>constant</td>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>36</th>
      <td>(8, 6)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>140</th>
      <td>(12, 2)</td>
      <td>tanh</td>
      <td>adaptive</td>
      <td>0.923810</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(3, 7)</td>
      <td>identity</td>
      <td>adaptive</td>
      <td>0.923810</td>
    </tr>
  </tbody>
</table>
</div>



### Análise de Desempenho

#### Baseado na Função de Ativação


```python
identity = analysis_dataset['mean_test_accuracy'][analysis_dataset['activation']=='identity']
logistic = analysis_dataset['mean_test_accuracy'][analysis_dataset['activation']=='logistic']
relu = analysis_dataset['mean_test_accuracy'][analysis_dataset['activation']=='relu']
tanh = analysis_dataset['mean_test_accuracy'][analysis_dataset['activation']=='tanh']
```


```python
func = [list(identity), list(logistic), list(relu), list(tanh)]
activations = ['Identidade', 'Sigmoidal', 'ReLU', 'Tanh']
index = []
for i in range(len(activations)):
    index.append(i+1)


plt.boxplot(func)
plt.title('Desempenho das RNAs para cada Função de Ativação testada\n')
plt.ylabel('Acurácia')

plt.xticks(index, activations)

plt.show()
```


![png](Atividade%20%234%20-%20RNA%20MLP_files/Atividade%20%234%20-%20RNA%20MLP_29_0.png)


Através do gráfico acima, é possível perceber que a Função de Ativação Identidade é a função que teve melhor desempenho nas RNAs testadas, visto que as acurácias das Redes com esta função se concentram em torno de 0,9.
Por outro lado, as funções ReLU e Tangente Hiperbólica se mostraram bastante heterogêneas, para o problema, em relação à acurácia.

#### Baseado no learning_rate


```python
adaptive = analysis_dataset['mean_test_accuracy'][analysis_dataset['learning_rate']=='adaptive']
constant = analysis_dataset['mean_test_accuracy'][analysis_dataset['learning_rate']=='constant']
```


```python
plt.plot(list(adaptive), label='adaptive')
plt.plot(list(constant), label = 'constant')
plt.legend()
plt.title('Desempenho das RNAs para cada Learning Rate testado')
plt.ylabel('Acurácia')
plt.xlabel('RNA')
plt.show()
```


![png](Atividade%20%234%20-%20RNA%20MLP_files/Atividade%20%234%20-%20RNA%20MLP_33_0.png)


O gráfico acima, indica que a escolha do learning_rate (adaptive ou constant) não exerce uma grande influência no desempenho da RNA, uma vez que as curvas de acurácia para as redes com cada um dos learning_rates são bem similares.
