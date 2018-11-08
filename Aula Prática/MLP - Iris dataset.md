
# Aula Prática - Desenvolvendo Redes Neurais Artificias com Dataset Iris

O objetivo desta aula consiste em conduzir um processo de Aprendizado de Máquina com Redes Neurais Artificias Multilayer Perceptron para o problema de classificação das flores Iris e analisar os resultados obtidos.

- Jean Phelipe de Oliveira Lima - 1615080096


## Bibliotecas


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from matplotlib import pyplot as plt
import numpy as np
```

## Leitura - Iris Dataset


```python
dataset = pd.read_csv('Iris.csv')
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
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## Holdout 80/20


```python
y = dataset['class']
x = dataset.drop(['class'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20)
```

# Redes Neurais Artificias Propostas:

## RNA #1
- Camadas ocultas: 1
- Número de Neurônios na(s) camada(s) oculta(s): 4
- Taxa de aprendizagem = 0,05
- Batch_size = 1
- Função de Ativação: Logistic


```python
MLP1 = MLPClassifier(hidden_layer_sizes=(4),
                     max_iter=50, 
                     activation='logistic', 
                     learning_rate='constant',
                     learning_rate_init=0.05,
                     batch_size = 1)
MLP1.fit(X_train, Y_train)

previsaoMLP1 = MLP1.predict(X_test)
print('Acurácia:', accuracy_score(Y_test, previsaoMLP1))
print('F1-Score:', f1_score(Y_test, previsaoMLP1, average='macro'))
```

    Acurácia: 0.5
    F1-Score: 0.48148148148148145


    /Users/jeanlima/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


## RNA #2
- Camadas ocultas: 1
- Número de Neurônios na(s) camada(s) oculta(s): 3
- Taxa de aprendizagem = 0,01
- Batch_size = 1
- Função de Ativação: ReLU


```python
MLP2 = MLPClassifier(hidden_layer_sizes=(3),
                     max_iter=50, 
                     activation='relu', 
                     learning_rate='constant',
                     learning_rate_init=0.01,
                     batch_size = 1)
MLP2.fit(X_train, Y_train)

previsaoMLP2 = MLP2.predict(X_test)
print('Acurácia:', accuracy_score(Y_test, previsaoMLP2))
print('F1-Score:', f1_score(Y_test, previsaoMLP2, average='macro'))
```

    Acurácia: 1.0
    F1-Score: 1.0


## RNA #3
- Camadas ocultas: 1
- Número de Neurônios na(s) camada(s) oculta(s): 5
- Taxa de aprendizagem = 0,01
- Batch_size = 1
- Função de Ativação: Logistic


```python
MLP3 = MLPClassifier(hidden_layer_sizes=(5),
                     max_iter=50, 
                     activation='logistic', 
                     learning_rate='constant',
                     learning_rate_init=0.01,
                     batch_size = 1)
MLP3.fit(X_train, Y_train)

previsaoMLP3 = MLP3.predict(X_test)
print('Acurácia:', accuracy_score(Y_test, previsaoMLP3))
print('F1-Score:', f1_score(Y_test, previsaoMLP3, average='macro'))
```

    Acurácia: 0.9666666666666667
    F1-Score: 0.9628647214854111


## RNA #4
- Camadas ocultas: 2
- Número de Neurônios na(s) camada(s) oculta(s): 3 - 4
- Taxa de aprendizagem = 0,01
- Batch_size = 1
- Função de Ativação: ReLU


```python
MLP4 = MLPClassifier(hidden_layer_sizes=(3,4),
                     max_iter=50, 
                     activation='relu', 
                     learning_rate='constant',
                     learning_rate_init=0.01,
                     batch_size = 1)
MLP4.fit(X_train, Y_train)

previsaoMLP4 = MLP4.predict(X_test)
print('Acurácia:', accuracy_score(Y_test, previsaoMLP4))
print('F1-Score:', f1_score(Y_test, previsaoMLP4, average='macro'))
```

    Acurácia: 1.0
    F1-Score: 1.0


## RNA #5
- Camadas ocultas: 2
- Número de Neurônios na(s) camada(s) oculta(s): 2 - 2
- Taxa de aprendizagem = 0,05
- Batch_size = 1
- Função de Ativação: Logistic


```python
MLP5 = MLPClassifier(hidden_layer_sizes=(2,2),
                     max_iter=50, 
                     activation='logistic', 
                     learning_rate='constant',
                     learning_rate_init=0.05,
                     batch_size = 1)
MLP5.fit(X_train, Y_train)

previsaoMLP5 = MLP5.predict(X_test)
print('Acurácia:', accuracy_score(Y_test, previsaoMLP5))
print('F1-Score:', f1_score(Y_test, previsaoMLP5, average='macro'))
```

    Acurácia: 0.2
    F1-Score: 0.11111111111111112


    /Users/jeanlima/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


## RNA #6
- Camadas ocultas: 2
- Número de Neurônios na(s) camada(s) oculta(s): 5 - 4
- Taxa de aprendizagem = 0,01
- Batch_size = 1
- Função de Ativação: ReLU


```python
MLP6 = MLPClassifier(hidden_layer_sizes=(5,4),
                     max_iter=50, 
                     activation='relu', 
                     learning_rate='constant',
                     learning_rate_init=0.01,
                     batch_size = 1)
MLP6.fit(X_train, Y_train)

previsaoMLP6 = MLP6.predict(X_test)
print('Acurácia:', accuracy_score(Y_test, previsaoMLP6))
print('F1-Score:', f1_score(Y_test, previsaoMLP6, average='macro'))
```

    Acurácia: 0.9666666666666667
    F1-Score: 0.9628647214854111


# Resultados


```python
fs1 = f1_score(Y_test, previsaoMLP1, average='macro')
fs2 = f1_score(Y_test, previsaoMLP2, average='macro')
fs3 = f1_score(Y_test, previsaoMLP3, average='macro')
fs4 = f1_score(Y_test, previsaoMLP4, average='macro')
fs5 = f1_score(Y_test, previsaoMLP5, average='macro')
fs6 = f1_score(Y_test, previsaoMLP6, average='macro')

previsoes = [previsaoMLP1,previsaoMLP2,previsaoMLP3,previsaoMLP4,previsaoMLP5,previsaoMLP6]
```

    /Users/jeanlima/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
lista_fs = []

lista_fs.append(fs1)
lista_fs.append(fs2)
lista_fs.append(fs3)
lista_fs.append(fs4)
lista_fs.append(fs5)
lista_fs.append(fs6)
```

As RNAs que tiveram melhor resultado, baseado no F1-Score, foram:


```python
maior = max(lista_fs)
lista_maiores=[]
for i in range(len(lista_fs)):
    if lista_fs[i] == maior:
        lista_maiores.append(i)

for i in lista_maiores:
    print('RNA #%d'%(i+1))
```

    RNA #2
    RNA #4


### Matrizes de Confusão


```python
for i in lista_maiores:
    print('RNA #%d:'%(i+1))
    print(confusion_matrix(Y_test,previsoes[i+1]))
    print()
```

    RNA #2:
    [[ 9  0  0]
     [ 0  6  0]
     [ 0  1 14]]
    
    RNA #4:
    [[ 0  9  0]
     [ 0  6  0]
     [ 0 15  0]]
    


### Distribuição das Médias do F1-Score para os dados dos testes


```python
mlps = ['MLP1', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLP6']
plt.title('Distribuição do F1-Score para os dados dos testes\n')
plt.plot(mlps, lista_fs)
plt.show()

lista_fs = np.array(lista_fs)
print('Parâmetros da Distribuição:\n')
print('Média:', lista_fs.mean())
print('Desvio Padrão:', lista_fs.std())
```


![png](MLP%20-%20Iris%20dataset_files/MLP%20-%20Iris%20dataset_28_0.png)


    Parâmetros da Distribuição:
    
    Média: 0.7530536725939024
    Desvio Padrão: 0.340550543713489


#### Pelos dados acima podemos constatar que o desempenho das 6 RNAs apresentadas foi heterogêneo.

### A RNA com mais neurônios ocultos é essencialmente a melhor?

Não. As RNAs com melhores resultados foram as #2 e #4, que possuiam, respectivamente, 3 e 7 neurônios ocultos.
Enquanto a RNA #6, com 9 neurônios ocultos, obteve desempenho inferior.

### As RNAs com uma única camada oculta tiveram F1-Score médio igual ou superior ao das redes com duas camadas ocultas? Isso ocorre em todo problema?


```python
print('Média do F1-Score para as RNAs com uma camada oculta:', lista_fs[:3].mean())
```

    Média do F1-Score para as RNAs com uma camada oculta: 0.8147820676556309



```python
print('Média do F1-Score para as RNAs com duas camadas ocultas:', lista_fs[3:].mean())
```

    Média do F1-Score para as RNAs com duas camadas ocultas: 0.6913252775321741


As RNAs com uma camada tiveram F1-Score médio superior ao F1-Score das RNAs com duas camadas ocultas.
No entanto, apenas o número de camadas não implica, diretamente, neste resultado, que varia de acordo com o problema, tanto que para o problema em questão, uma das duas RNAs que obtiveram o melhor resultado possuia uma camada oculta e outra posuia duas camadas ocultas.
