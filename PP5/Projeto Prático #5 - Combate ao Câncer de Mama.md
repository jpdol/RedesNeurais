
# Projeto Prático #5

O PP5 trata-se de uma [competição](https://www.kaggle.com/c/combatendo-o-cncer-de-mama), entre os alunos da disciplina, na plataforma [Kaggle](https://www.kaggle.com).

Neste projeto prático, as equipes terão de identificar o papel de marcadores biológicos na presença ou ausência de câncer de mama. O câncer de mama é o tipo de câncer mais comum entre as mulheres no mundo e no Brasil, depois do de pele não melanoma, respondendo por cerca de 28% dos casos novos a cada ano. O câncer de mama também acomete homens, porém é raro, representando apenas 1% do total de casos da doença. Para o ano de 2018 foram estimados 60 mil novos casos da doença, conforme [INCA](http://www2.inca.gov.br/wps/wcm/connect/tiposdecancer/site/home/mama).

Pesquisadores da Universidade de Coimbra obtiveram 10 preditores quantitativos correspondentes a dados antropométricos de pacientes, todos oriundos de exames de sangue de rotina. Se modelos inteligentes baseados nestes preditores forem acurados, há potencial para uso destes biomarcadores como indicador de câncer de mama. Leia mais sobre em [UCI Breast Cancer Coimbra](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra).

- Aluno: Jean Phelipe de Oliveira Lima. Matrícula: 1615080096

## Bibliotecas


```python
import pandas as pd
import numpy as np
from math import ceil
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
```

## Breast Cancer Coimbra Dataset - Conjunto de treino


```python
dataset_with_id = pd.read_csv('train.csv')
dataset = dataset_with_id.drop(['id'], axis=1)
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
      <th>Age</th>
      <th>BMI</th>
      <th>Glucose</th>
      <th>Insulin</th>
      <th>HOMA</th>
      <th>Leptin</th>
      <th>Adiponectin</th>
      <th>Resistin</th>
      <th>MCP.1</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47</td>
      <td>22.0300</td>
      <td>84</td>
      <td>2.869</td>
      <td>0.5900</td>
      <td>26.6500</td>
      <td>38.0400</td>
      <td>3.3200</td>
      <td>191.720</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75</td>
      <td>30.4800</td>
      <td>152</td>
      <td>7.010</td>
      <td>2.6283</td>
      <td>50.5300</td>
      <td>10.0600</td>
      <td>11.7300</td>
      <td>99.450</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>22.8600</td>
      <td>82</td>
      <td>4.090</td>
      <td>0.8273</td>
      <td>20.4500</td>
      <td>23.6700</td>
      <td>5.1400</td>
      <td>313.730</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54</td>
      <td>24.2188</td>
      <td>86</td>
      <td>3.730</td>
      <td>0.7913</td>
      <td>8.6874</td>
      <td>3.7052</td>
      <td>10.3446</td>
      <td>635.049</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69</td>
      <td>35.0927</td>
      <td>101</td>
      <td>5.646</td>
      <td>1.4066</td>
      <td>83.4821</td>
      <td>6.7970</td>
      <td>82.1000</td>
      <td>263.499</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
def piramide_geometrica(ni, no, alfa):
    nh = alfa*((ni*no)**(1/2))
    return ceil(nh)
```


```python
def hidden_layers(layers, nh):
    for i in range(1, nh):
        neurons_layers = (i, nh-i)
        layers.append(neurons_layers)
    return layers
```


```python
num_in = 9
num_out = 1
alpha = [2, 3]
layers = []

for i in range(len(alpha)):
    nh = piramide_geometrica(num_in, num_out, alpha[i])
    print('Para α = %.1f, Nh = %d'%(alpha[i],nh))
    hidden_layers(layers, nh)#insere cada possibilidade de camadas ocultas, dado o numero de neurônios, na lista 'layers'
    
print()
print('Distribuições de Camadas Ocultas:\n')
for i in layers:
    print(i)
```

    Para α = 2.0, Nh = 6
    Para α = 3.0, Nh = 9
    
    Distribuições de Camadas Ocultas:
    
    (1, 5)
    (2, 4)
    (3, 3)
    (4, 2)
    (5, 1)
    (1, 8)
    (2, 7)
    (3, 6)
    (4, 5)
    (5, 4)
    (6, 3)
    (7, 2)
    (8, 1)


### Treinamento de Redes Neurais Artificiais do tipo Multilayer Perceptron

- Utilização de busca em grade para encontrar melhores parâmetros e hiperparâmetros para a rede.
- Validação cruzada com k = 5 folds


```python
parameters = {'solver': ['lbfgs'], 
              'activation': ['identity'],
              'hidden_layer_sizes': layers,
              'max_iter':[1000],
              'batch_size': [16, 32]}

# Busca em grade e validação cruzada. K=5 folds
gs = GridSearchCV(MLPClassifier(), 
                  parameters, 
                  cv=5, 
                  scoring='accuracy')
```


```python
#Atributos preditores
x = dataset.drop(['Classification'], axis = 1) 

#Atributo Alvo
y = dataset.Classification 
```


```python
gs.fit(x, y)
```




    GridSearchCV(cv=5, error_score='raise-deprecating',
           estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=None, shuffle=True, solver='adam', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'solver': ['lbfgs'], 'activation': ['identity'], 'hidden_layer_sizes': [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (1, 8), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1)], 'max_iter': [1000], 'batch_size': [16, 32]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)



### Resultados do GridSerach

#### 7 melhores RNAs da busca


```python
results = pd.DataFrame(gs.cv_results_)
analysis_dict = {}

analysis_dict['hidden_layer_sizes'] = results['param_hidden_layer_sizes']
analysis_dict['activation'] = results['param_activation']
analysis_dict['max_iter'] = results['param_max_iter']
analysis_dict['batch_size'] = results['param_batch_size']
analysis_dict['mean_test_accuracy'] = results['mean_test_score']

analysis_dataset = pd.DataFrame(analysis_dict)
top7 = analysis_dataset.sort_values('mean_test_accuracy', ascending=False).head(7)
top7
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
      <th>max_iter</th>
      <th>batch_size</th>
      <th>mean_test_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>(2, 7)</td>
      <td>identity</td>
      <td>1000</td>
      <td>32</td>
      <td>0.717391</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(1, 8)</td>
      <td>identity</td>
      <td>1000</td>
      <td>32</td>
      <td>0.673913</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(3, 6)</td>
      <td>identity</td>
      <td>1000</td>
      <td>16</td>
      <td>0.673913</td>
    </tr>
    <tr>
      <th>22</th>
      <td>(5, 4)</td>
      <td>identity</td>
      <td>1000</td>
      <td>32</td>
      <td>0.663043</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(1, 5)</td>
      <td>identity</td>
      <td>1000</td>
      <td>32</td>
      <td>0.652174</td>
    </tr>
    <tr>
      <th>23</th>
      <td>(6, 3)</td>
      <td>identity</td>
      <td>1000</td>
      <td>32</td>
      <td>0.652174</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(4, 2)</td>
      <td>identity</td>
      <td>1000</td>
      <td>32</td>
      <td>0.652174</td>
    </tr>
  </tbody>
</table>
</div>



## Solução #1 

Na primeira solução, uma entrada será submetida às 7 melhores redes neurais obtidas por meio da busca em grade.
Será escolhida a resposta que mais se repete dentre as saídas das 7 redes.


```python
top7_matrix=[]
for i in top7.index:
    top7_matrix.append(list(top7.loc[i]))

mlps = []
for i in top7_matrix:
    mlps.append(MLPClassifier(hidden_layer_sizes = i[0], 
                     activation = i[1], 
                     max_iter=i[2], 
                     solver = 'lbfgs',
                     batch_size = i[3]))
    
for i in range(len(mlps)):
    mlps[i].fit(x, y)
```


```python
# Função para verificar qual saída se repete mais
def vote(classes, predict):
    winner = classes[0]
    for i in classes:
        if predict.count(i)> predict.count(winner):
            winner = i
    return winner
```


```python
# Função para que define a Solução 1
def predict_mlps_winner(data, mlps):
    predicts = []
    for i in range(len(mlps)):
        predicts.append(mlps[i].predict([data]))
    return vote([1,2], predicts)
```

### Conjunto de Teste


```python
testes_with_id = pd.read_csv('test.csv')
testes = testes_with_id.drop(['id'], axis = 1)
testes.head()
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
      <th>Age</th>
      <th>BMI</th>
      <th>Glucose</th>
      <th>Insulin</th>
      <th>HOMA</th>
      <th>Leptin</th>
      <th>Adiponectin</th>
      <th>Resistin</th>
      <th>MCP.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62</td>
      <td>22.6562</td>
      <td>92</td>
      <td>3.482</td>
      <td>0.7902</td>
      <td>9.8648</td>
      <td>11.2362</td>
      <td>10.6955</td>
      <td>703.973</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29</td>
      <td>23.0100</td>
      <td>82</td>
      <td>5.663</td>
      <td>1.1454</td>
      <td>35.5900</td>
      <td>26.7200</td>
      <td>4.5800</td>
      <td>174.800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75</td>
      <td>25.7000</td>
      <td>94</td>
      <td>8.079</td>
      <td>1.8733</td>
      <td>65.9260</td>
      <td>3.7412</td>
      <td>4.4968</td>
      <td>206.802</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
      <td>27.8876</td>
      <td>99</td>
      <td>9.208</td>
      <td>2.2486</td>
      <td>12.6757</td>
      <td>5.4782</td>
      <td>23.0331</td>
      <td>407.206</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75</td>
      <td>23.0000</td>
      <td>83</td>
      <td>4.952</td>
      <td>1.0138</td>
      <td>17.1270</td>
      <td>11.5790</td>
      <td>7.0913</td>
      <td>318.302</td>
    </tr>
  </tbody>
</table>
</div>



### Testes para a Solução #1


```python
results = []
for i in range(len(testes)):
    results.append(predict_mlps_winner(testes.loc[i], mlps))
```


```python
dict_results = {'id': list(testes_with_id.id),'Classification': results}
submission = pd.DataFrame(dict_results)
submission.head()
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
      <th>id</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>113</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>86</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv('submission.csv', index=False)
```

## Solução #2

Esta solução consiste na implementação de um *ensemble* . As saídas das 7 melhores RNAs obtidas pela busca em grade, serão submetidas à outra rede neural, esta decidirá a resposta final.

### Novo conjunto de treino
Saídas 7 melhores RNAs para cada instância do dataset original.


```python
new_dataset = []
for i in range(len(mlps)):
    new_dataset.append(mlps[i].predict(x))
new_dataset.append(np.asarray(y))
```


```python
new_dataset = np.asarray(new_dataset)
new_dataset = pd.DataFrame(new_dataset.transpose(), columns=[1,2,3,4,5,6,7,'Classification'])
new_dataset.head()
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Função que define a solção #2
def ensemble_winner(data, mlps, ensemble):
    predicts = []
    for i in range(len(mlps)):
        predicts.append(mlps[i].predict([data]))
    predicts = np.asarray(predicts)
    return ensemble.predict(predicts.transpose())
```

### Treino - *ensemble*


```python
ensemble = MLPClassifier(hidden_layer_sizes=(14,7),
                         activation= 'relu',
                         batch_size = 3,
                         max_iter = 1000,
                         learning_rate = 'constant',
                         learning_rate_init = 0.0005)

#Atributos preditores do novo dataset
new_x = new_dataset.drop(['Classification'], axis=1)
#Atributo alvo do novo dataset
new_y = new_dataset.Classification

#Treino do ensemble
ensemble.fit(new_x, new_y)
```




    MLPClassifier(activation='relu', alpha=0.0001, batch_size=3, beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(14, 7), learning_rate='constant',
           learning_rate_init=0.0005, max_iter=1000, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=None, shuffle=True, solver='adam', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False)



### Testes para a Solução #2


```python
ensemble_results = []

for i in range(len(testes)):
    ensemble_results.append(ensemble_winner(testes.loc[i], mlps, ensemble))
```


```python
results = []
for i in ensemble_results:
    results.append(i[0])
dict_results = {'id': list(testes_with_id.id),'Classification': results}
submission = pd.DataFrame(dict_results)
submission.head()
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
      <th>id</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>113</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>86</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv('submission.csv', index=False)
```

## Solução #3

Esta solução também consiste em um ensemble. No entanto, desta vez, as 7 melhores redes neurais, obtidas na busca em grade, fornecerão as probabilidades de previsão (*predict_proba*) como entrada para uma nova RNA que decidirá a resposta final.

### Construção do novo dataset


```python
def proba_into_list(proba, proba_list_1, proba_list_2):
    for i in range(len(proba)):
        proba_list_1.append(proba[i][0])
        proba_list_2.append(proba[i][1])

rna1_prob1 = []
rna1_prob2 = []
rna2_prob1 = []
rna2_prob2 = []
rna3_prob1 = []
rna3_prob2 = []
rna4_prob1 = []
rna4_prob2 = []
rna5_prob1 = []
rna5_prob2 = []
rna6_prob1 = []
rna6_prob2 = []
rna7_prob1 = []
rna7_prob2 = []
rna8_prob1 = []
rna8_prob2 = []
rna9_prob1 = []
rna9_prob2 = []

proba = mlps[0].predict_proba(x)
proba_into_list(proba, rna1_prob1, rna1_prob2)
proba = mlps[1].predict_proba(x)
proba_into_list(proba, rna2_prob1, rna2_prob2)
proba = mlps[2].predict_proba(x)
proba_into_list(proba, rna3_prob1, rna3_prob2)
proba = mlps[3].predict_proba(x)
proba_into_list(proba, rna4_prob1, rna4_prob2)
proba = mlps[4].predict_proba(x)
proba_into_list(proba, rna5_prob1, rna5_prob2)
proba = mlps[5].predict_proba(x)
proba_into_list(proba, rna6_prob1, rna6_prob2)
proba = mlps[6].predict_proba(x)
proba_into_list(proba, rna7_prob1, rna7_prob2)

    
classification = np.asarray(y)

proba_dataset = {'rna1_prob1':rna1_prob1, 
                 'rna1_prob2':rna1_prob2, 
                 'rna2_prob1':rna2_prob1,
                 'rna2_prob2':rna2_prob2,
                 'rna3_prob1':rna3_prob1,
                 'rna3_prob2':rna3_prob2,
                 'rna4_prob1':rna4_prob1, 
                 'rna4_prob2':rna4_prob2, 
                 'rna5_prob1':rna5_prob1,
                 'rna5_prob2':rna5_prob2,
                 'rna6_prob1':rna6_prob1,
                 'rna6_prob2':rna6_prob2,
                 'rna7_prob1':rna7_prob1, 
                 'rna7_prob2':rna7_prob2, 
                 'Classification':classification}

proba_dataset = pd.DataFrame(proba_dataset)
proba_dataset.head(5)
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
      <th>rna1_prob1</th>
      <th>rna1_prob2</th>
      <th>rna2_prob1</th>
      <th>rna2_prob2</th>
      <th>rna3_prob1</th>
      <th>rna3_prob2</th>
      <th>rna4_prob1</th>
      <th>rna4_prob2</th>
      <th>rna5_prob1</th>
      <th>rna5_prob2</th>
      <th>rna6_prob1</th>
      <th>rna6_prob2</th>
      <th>rna7_prob1</th>
      <th>rna7_prob2</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.686067</td>
      <td>0.313933</td>
      <td>0.477146</td>
      <td>0.522854</td>
      <td>1.00000</td>
      <td>2.912713e-15</td>
      <td>0.694470</td>
      <td>0.305530</td>
      <td>0.705509</td>
      <td>0.294491</td>
      <td>0.675457</td>
      <td>0.324543</td>
      <td>0.698050</td>
      <td>0.301950</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.033584</td>
      <td>0.966416</td>
      <td>0.477712</td>
      <td>0.522288</td>
      <td>1.00000</td>
      <td>7.408376e-44</td>
      <td>0.031571</td>
      <td>0.968429</td>
      <td>0.032888</td>
      <td>0.967112</td>
      <td>0.031898</td>
      <td>0.968102</td>
      <td>0.031097</td>
      <td>0.968903</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.613089</td>
      <td>0.386911</td>
      <td>0.471049</td>
      <td>0.528951</td>
      <td>0.00001</td>
      <td>9.999896e-01</td>
      <td>0.618909</td>
      <td>0.381091</td>
      <td>0.624653</td>
      <td>0.375347</td>
      <td>0.600535</td>
      <td>0.399465</td>
      <td>0.619025</td>
      <td>0.380975</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.568511</td>
      <td>0.431489</td>
      <td>0.453600</td>
      <td>0.546400</td>
      <td>0.00000</td>
      <td>1.000000e+00</td>
      <td>0.571764</td>
      <td>0.428236</td>
      <td>0.570457</td>
      <td>0.429543</td>
      <td>0.573582</td>
      <td>0.426418</td>
      <td>0.559524</td>
      <td>0.440476</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.183284</td>
      <td>0.816716</td>
      <td>0.470458</td>
      <td>0.529542</td>
      <td>1.00000</td>
      <td>2.252961e-12</td>
      <td>0.157231</td>
      <td>0.842769</td>
      <td>0.165420</td>
      <td>0.834580</td>
      <td>0.173593</td>
      <td>0.826407</td>
      <td>0.147002</td>
      <td>0.852998</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Conjunto de Teste


```python
rna1_prob1 = []
rna1_prob2 = []
rna2_prob1 = []
rna2_prob2 = []
rna3_prob1 = []
rna3_prob2 = []
rna4_prob1 = []
rna4_prob2 = []
rna5_prob1 = []
rna5_prob2 = []
rna6_prob1 = []
rna6_prob2 = []
rna7_prob1 = []
rna7_prob2 = []


proba = mlps[0].predict_proba(testes)
proba_into_list(proba, rna1_prob1, rna1_prob2)
proba = mlps[1].predict_proba(testes)
proba_into_list(proba, rna2_prob1, rna2_prob2)
proba = mlps[2].predict_proba(testes)
proba_into_list(proba, rna3_prob1, rna3_prob2)
proba = mlps[3].predict_proba(testes)
proba_into_list(proba, rna4_prob1, rna4_prob2)
proba = mlps[4].predict_proba(testes)
proba_into_list(proba, rna5_prob1, rna5_prob2)
proba = mlps[5].predict_proba(testes)
proba_into_list(proba, rna6_prob1, rna6_prob2)
proba = mlps[6].predict_proba(testes)
proba_into_list(proba, rna7_prob1, rna7_prob2)
    
test_proba = {}
test_proba = {'rna1_prob1':rna1_prob1, 
                 'rna1_prob2':rna1_prob2, 
                 'rna2_prob1':rna2_prob1,
                 'rna2_prob2':rna2_prob2,
                 'rna3_prob1':rna3_prob1,
                 'rna3_prob2':rna3_prob2,
                 'rna4_prob1':rna4_prob1, 
                 'rna4_prob2':rna4_prob2, 
                 'rna5_prob1':rna5_prob1,
                 'rna5_prob2':rna5_prob2,
                 'rna6_prob1':rna6_prob1,
                 'rna6_prob2':rna6_prob2,
                 'rna7_prob1':rna7_prob1, 
                 'rna7_prob2':rna7_prob2, 
}

test_proba = pd.DataFrame(test_proba)
test_proba.head(5)
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
      <th>rna1_prob1</th>
      <th>rna1_prob2</th>
      <th>rna2_prob1</th>
      <th>rna2_prob2</th>
      <th>rna3_prob1</th>
      <th>rna3_prob2</th>
      <th>rna4_prob1</th>
      <th>rna4_prob2</th>
      <th>rna5_prob1</th>
      <th>rna5_prob2</th>
      <th>rna6_prob1</th>
      <th>rna6_prob2</th>
      <th>rna7_prob1</th>
      <th>rna7_prob2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.424908</td>
      <td>0.575092</td>
      <td>0.449497</td>
      <td>0.550503</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.428411</td>
      <td>0.571589</td>
      <td>0.431871</td>
      <td>0.568129</td>
      <td>0.427740</td>
      <td>0.572260</td>
      <td>0.412313</td>
      <td>0.587687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.660227</td>
      <td>0.339773</td>
      <td>0.478272</td>
      <td>0.521728</td>
      <td>1.000000e+00</td>
      <td>2.424616e-14</td>
      <td>0.665328</td>
      <td>0.334672</td>
      <td>0.670174</td>
      <td>0.329826</td>
      <td>0.637744</td>
      <td>0.362256</td>
      <td>0.666276</td>
      <td>0.333724</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.697612</td>
      <td>0.302388</td>
      <td>0.475358</td>
      <td>0.524642</td>
      <td>1.000000e+00</td>
      <td>9.448002e-24</td>
      <td>0.696919</td>
      <td>0.303081</td>
      <td>0.692082</td>
      <td>0.307918</td>
      <td>0.662993</td>
      <td>0.337007</td>
      <td>0.685389</td>
      <td>0.314611</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.242636</td>
      <td>0.757364</td>
      <td>0.464623</td>
      <td>0.535377</td>
      <td>2.642331e-14</td>
      <td>1.000000e+00</td>
      <td>0.238947</td>
      <td>0.761053</td>
      <td>0.238400</td>
      <td>0.761600</td>
      <td>0.253605</td>
      <td>0.746395</td>
      <td>0.237756</td>
      <td>0.762244</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.724633</td>
      <td>0.275367</td>
      <td>0.470275</td>
      <td>0.529725</td>
      <td>8.136776e-01</td>
      <td>1.863224e-01</td>
      <td>0.731350</td>
      <td>0.268650</td>
      <td>0.730083</td>
      <td>0.269917</td>
      <td>0.728275</td>
      <td>0.271725</td>
      <td>0.724326</td>
      <td>0.275674</td>
    </tr>
  </tbody>
</table>
</div>



### Treino


```python
proba_ensemble = MLPClassifier(hidden_layer_sizes=(13,20),
                         activation= 'relu',
                         batch_size = 3,
                         max_iter = 10000,
                         learning_rate = 'constant',
                         learning_rate_init = 0.00005)

#Atributos preditores do novo dataset
x_proba = proba_dataset.drop(['Classification'], axis=1)
#Atributo alvo do novo dataset
y_proba = proba_dataset.Classification

#Treino do ensemble
proba_ensemble.fit(x_proba, y_proba)
```




    MLPClassifier(activation='relu', alpha=0.0001, batch_size=3, beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(13, 20), learning_rate='constant',
           learning_rate_init=5e-05, max_iter=10000, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=None, shuffle=True, solver='adam', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False)



### Testes


```python
dict_results = {'id': list(testes_with_id.id),'Classification': proba_ensemble_results}
submission = pd.DataFrame(dict_results)
submission.head()
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
      <th>id</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>113</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>86</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv('submission.csv', index=False)
```
