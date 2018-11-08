
# Teste 2 - 11/09/2018

## Disponibilização: 10/09/2018 - 11h
## Encerramento: 11/09/2018 - 20h

O objetivo deste segundo projeto prático da disciplina Redes Neurais Artificias é praticar os conceitos de Machine Learning vistos até o momento, em especial aqueles relativos ao processo de Aprendizagem de Máquina.

Vamos trabalhar com o dataset **Breast Cancer Wisconsin (Diagnostic) Data Set**, vide: <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)">Repositório UCI</a>

Esta tarefa é dividida em to-dos, isto é, pequenas atividades que devem ser cumpridas para que o objetivo geral seja alcançado. A cada to-do está associada uma célula do Jupyter Notebook, que deve ser preenchida com código Python atendendo ao que se pede.


Edite aqui o nome da equipe:

- Jean Phelipe de Oliveira Lima - 1615080096
- Rodrigo Gomes de Souza - 1715310022


```python
# Organize seus imports nesta célula
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
```

### To-Do 1

1. Você deve importar o dataset a partir do sci-kit learn.
Consulte o link: [Link da documentação do sci-kit learn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer)
   * Este dataset está organizado sob a forma de um dicionário, em que os dados preditores encontram-se na chave 'data', composta de diversas matrizes. Cada matriz está associada a um nome 'feature_names'. 
2. Crie um novo dicionário que mapeia cada 'feature_name' para uma matriz correspondente.
    * Antes de fazer esta associação, transponha a matriz localizada na chave 'data' para obter a dimensão correta.
3. Transforme o dataset em um objetivo tipo DataFrame do pandas
4. Adicione o atributo-alvo ao dataset existente.
    * Importante: O atributo-alvo está na chave 'target' do dicionário, com nome 'target_names'


```python
#1 - Importando o dataset
data = load_breast_cancer()

#2 - Criando dicionário para mapear cada feature_name
data_transp = np.transpose(data.data)
data_dict = {}
for i in range(len(data.feature_names)):
    data_dict[data.feature_names[i]] = data_transp[i]

#3 - Convertendo em um DataFrame do pandas
dataset = pd.DataFrame.from_dict(data_dict)


#4 - Adicionando o atributo alvo
target_array = []
for i in range(len(data.target)):
    target_array.append(data.target_names[data.target[i]])
dataset['target'] = pd.Series(target_array)
```

### To-Do 2

Utilizando `pandas.DataFrame` para manipular o dataset, faça o que se pede:
1. Informe a quantidade de exemplos existentes no dataset
2. Enumere os atributos existentes no dataset
3. Identifique o atributo-alvo e imprima-o
4. O dataset é balanceado?
5. Remova todos os atributos que contenham a palavra `error`


```python
#1 - Quantidade de Exemplos
len(dataset)
```




    569




```python
#2 - Colunas do dataset
for i in dataset.columns:
    print (i)
```

    mean radius
    mean texture
    mean perimeter
    mean area
    mean smoothness
    mean compactness
    mean concavity
    mean concave points
    mean symmetry
    mean fractal dimension
    radius error
    texture error
    perimeter error
    area error
    smoothness error
    compactness error
    concavity error
    concave points error
    symmetry error
    fractal dimension error
    worst radius
    worst texture
    worst perimeter
    worst area
    worst smoothness
    worst compactness
    worst concavity
    worst concave points
    worst symmetry
    worst fractal dimension
    target



```python
#3 - Atributo alvo
dataset.target
```




    0      malignant
    1      malignant
    2      malignant
    3      malignant
    4      malignant
    5      malignant
    6      malignant
    7      malignant
    8      malignant
    9      malignant
    10     malignant
    11     malignant
    12     malignant
    13     malignant
    14     malignant
    15     malignant
    16     malignant
    17     malignant
    18     malignant
    19        benign
    20        benign
    21        benign
    22     malignant
    23     malignant
    24     malignant
    25     malignant
    26     malignant
    27     malignant
    28     malignant
    29     malignant
             ...    
    539       benign
    540       benign
    541       benign
    542       benign
    543       benign
    544       benign
    545       benign
    546       benign
    547       benign
    548       benign
    549       benign
    550       benign
    551       benign
    552       benign
    553       benign
    554       benign
    555       benign
    556       benign
    557       benign
    558       benign
    559       benign
    560       benign
    561       benign
    562    malignant
    563    malignant
    564    malignant
    565    malignant
    566    malignant
    567    malignant
    568       benign
    Name: target, Length: 569, dtype: object




```python
#4 - Retornando True se dataset balanceado, isto é, a ocorrência de uma classe não é maior que 5% em relacao a outra,
#    Retornando False se não balanceado.
benignos = dataset['target'][dataset['target']=='benign'].count()
malignos = dataset['target'][dataset['target']=='malignant'].count()

if benignos > malignos + (malignos*5)/100:
    balanceado = False
elif malignos > benignos + (malignos*5)/100:
    balanceado = False
else:
    balanceado = True

balanceado
```




    False




```python
#5 - Removendo todos os atributos que contenham a palavra 'error'
for i in dataset.columns:
    if 'error' in dataset[i]:
        dataset.drop([i], axis=1, inplace = True)
```

### To-Do 3

Faça uma partição randomizada do tipo 70/30 para conjunto de treinamento e de testes.
Em ambos os conjuntos, separe o atributo-alvo.

Para facilitar, siga a nomenclatura sugerida:
* X_train: atributos preditores para o conjunto de treinamento
* X_test: atributos preditores para o conjunto de testes
* Y_train: atributo-alvo para os exemplos do conjunto de treinamento
* Y_test: atributo-alvo para os exemplos do conjunto de testes

Sugestão: [consultar a documentação do sci-kit learn](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)



```python
y = dataset['target']
dataset.drop(['target'], axis=1, inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(dataset, y, test_size=0.30)
```

### To-Do 4

Vamos usar os dados X_train e Y_train para treinar dois modelos diferentes de Aprendizagem de Máquina.
1. Modelo 1: Vizinhos mais próximos, com k = 5
2. Modelo 2: Centróides mais próximos, de acordo com a distância Euclidiana


```python
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# 5 - vizinhos mais próximos
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

# Kernel Density
nc = NearestCentroid()
nc.fit(X_train, Y_train)
```




    NearestCentroid(metric='euclidean', shrink_threshold=None)



### To-Do 5

Utilizar o conjunto de testes para prever o conjunto de testes


```python
previsaokNN = knn.predict(X_test)
previsaonc = nc.predict(X_test)
```

### To-Do 6

Analisando as diferenças e igualdades entre os vetores previsaokNN, previsaonc e Y_test, construa as matrizes de confusão dos respectivos modelos de Machine Learning. 

Consulte: [Documentação do sklearn para Matrizes de Confusão](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)


```python
print(confusion_matrix(Y_test, previsaokNN))
```

    [[110   4]
     [  8  49]]



```python
print(confusion_matrix(Y_test, previsaonc))
```

    [[113   1]
     [ 17  40]]


### To-Do 7

Para cada um dos modelos, apresente:

1. Acurácia
2. Precisão
3. Revocação
4. F-Score (Leve em consideração se o dataset é balanceado ou não)

Consulte: [Documentação do sklearn para Métricas de Desempenho](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)


```python
#1.1 - Acurácia para modelo 5-Vizinhos mais próximos
accuracy_score(Y_test, previsaokNN)
```




    0.9298245614035088




```python
#1.2 - Acurácia para modelo Centróides mais próximos
accuracy_score(Y_test, previsaonc)
```




    0.8947368421052632




```python
#2.1 - Precisão para modelo 5-Vizinhos mais próximos
precision_score(Y_test, previsaokNN, pos_label='benign')
```




    0.9322033898305084




```python
#3.2 - Precisão para modelo Centróides mais próximos
precision_score(Y_test, previsaonc, pos_label='benign')
```




    0.8692307692307693




```python
#4.1 - Revocação para modelo 5-Vizinhos mais próximos 
recall_score(Y_test, previsaokNN, pos_label='benign')
```




    0.9649122807017544




```python
#4.2 - Revocação para modelo Centróides mais próximos
recall_score(Y_test, previsaonc, pos_label='benign')
```




    0.9912280701754386




```python
#5.1 - f1-score para modelo 5-Vizinhos mais próximos
f1_score(Y_test, previsaokNN, pos_label='benign')
```




    0.9482758620689654




```python
#5.2 - f1-score para modelo Centróides mais próximos
f1_score(Y_test, previsaonc, pos_label='benign')
```




    0.9262295081967212



### To-Do 8

Utilizando argumentos textuais, justifique qual dos modelos apresentados é melhor para o problema em questão.

O melhor modelo apresentado para o problema é o modelo 5-Vizinhos mais próximos, tendo em vista que, dentre as métricas apresentadas, apenas a revocação teve um valor menor quando comparado ao modelo Centróides mais próximos.

Além disso, pela matriz de confusão, também podemos perceber que o modelo Centróides mais próximos apresenta um maior número de calssificações onde um paciente com câncer malígno fora diagnosticado com câncer benígno, o que é o grave erro para o contexto em que o modelo deve ser empregado.
