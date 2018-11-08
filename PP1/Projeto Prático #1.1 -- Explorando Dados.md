
# Projeto Prático 1.1 -- Explorando Dados

O projeto prático 1.1 da disciplina de Redes Neurais Artificiais deve ser desenvolvido em duplas justas. A ideia geral é seguir o passo a passo das atividades solicitadas para aprender a utilizar as bibliotecas Python para praticar os conceitos de exploração de dados vistos ao longo dessas primeiras aulas.

Na avaliação será levado em conta:
1. Corretude das tarefas solicitadas
2. Qualidade e boas práticas de codificação
3. Eficiência na manipulação dos dados


Preecha aqui os integrantes da dupla e suas respectivas matrículas (duplo clique para editar):
- Jean Phelipe de Oliveira Lima - 1615080096

## Apresentação da Atividade

Vamos aprender um pouco mais sobre carros! Para tanto, vamos utilizar o [dataset AutoMPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg), que contém informações de veículos do ano de 1983 apresentados em uma exposição nos EUA. O atributo alvo chama-se mpg, denotando milhas por galão, uma unidade de medida equivalente ao nosso quilômetro por litro.


```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

## Tarefa 1: Abrir o dataset

1. Baixe o arquivo 'autompg.csv' do Google Classroom e o abra com a biblioteca pandas
2. Imprima o cabeçalho do dataset
3. Imprima os tipos de dados no dataset


```python
# 1 - Abrindo o dataset
dataset = pd.read_csv('autompg.csv', encoding='UTF-8', sep=';')
```


```python
# 2 - Imprimindo o cabeçalho
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693.0</td>
      <td>11.5</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436.0</td>
      <td>11.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8.0</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8.0</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449.0</td>
      <td>10.5</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 3 - Imprimindo os tipos de dados presentes no dataset
dataset.dtypes
```




    mpg             float64
    cylinders       float64
    displacement    float64
    horsepower      float64
    weight          float64
    acceleration    float64
    modelyear       float64
    origin          float64
    name             object
    dtype: object



## Tarefa 2: Conhecendo e limpando os dados

1. Imprima os nomes das colunas do dataset
2. Quantos exemplos o dataset possui?
3. Quantos exemplos com dados faltantes o dataset possui?
4. Efetue a limpeza do dataset excluindo todos os exemplos em que há alguma informação faltando. Daqui em diante, considere essa versão do dataset.


```python
# 1 - Imprimindo nomes das colunas:
columns = dataset.columns
for column in columns:
    print(column)
```

    mpg
    cylinders
    displacement
    horsepower
    weight
    acceleration
    modelyear
    origin
    name



```python
# 2 - Imprimindo o número de exemplos do dataset
len(dataset)
```




    406




```python
# 3 - Imprimindo o número de exemplos com dados faltantes
cont=0
for i in range(len(dataset)):
    linha = dataset.loc[i].isnull()
    contou = False
    for j in linha:
        if j and not contou:
            cont+=1
            contou = True
            
print(cont)
```

    14



```python
# 4 - Excluindo exemplos com dados faltantes
dataset.dropna(inplace=True)
```

## Tarefa 3: Consulta aos dados

1. Calcule a média do atributo alvo mpg
2. Imprima o nome dos carros cujo consumo (mpg) é maior que a média
3. Qual o carro mais eficiente (maior mpg)?
4. Quantos carros foram fabricados após 1977?
5. Qual a cilindrada média dos carros fabricados entre 1980 e 1982?
6. Há quantos carros da marca Chevrolet no dataset? Imprima todas as características dos mesmos.


```python
# 1 - Imprimindo média dos valores do atributo 'mpg'
mean = dataset['mpg'].mean()
mean
```




    23.44591836734694




```python
# 2 - Imprimindo nome dos carros com 'mpg' maior que a média
dataset['name'][dataset['mpg'] >= mean]
```




    20                 toyota corona mark ii
    24                          datsun pl510
    25          volkswagen 1131 deluxe sedan
    26                           peugeot 504
    27                           audi 100 ls
    28                              saab 99e
    29                              bmw 2002
    35                          datsun pl510
    36                   chevrolet vega 2300
    37                         toyota corona
    57                             opel 1900
    58                           peugeot 304
    59                             fiat 124b
    60                   toyota corolla 1200
    61                           datsun 1200
    62                  volkswagen model 111
    63                      plymouth cricket
    64                 toyota corona hardtop
    65                    dodge colt hardtop
    86                       renault 12 (sw)
    88                       datsun 510 (sw)
    90                       dodge colt (sw)
    91              toyota corolla 1600 (sw)
    109              volkswagen super beetle
    121                 fiat 124 sport coupe
    124                             fiat 128
    125                           opel manta
    129                            saab 99le
    136                          datsun b210
    137                           ford pinto
                         ...                
    372                oldsmobile cutlass ls
    375                   chevrolet cavalier
    376             chevrolet cavalier wagon
    377            chevrolet cavalier 2-door
    378           pontiac j2000 se hatchback
    379                       dodge aries se
    380                      pontiac phoenix
    381                 ford fairmont futura
    383                  volkswagen rabbit l
    384                   mazda glc custom l
    385                     mazda glc custom
    386               plymouth horizon miser
    387                       mercury lynx l
    388                     nissan stanza xe
    389                         honda accord
    390                       toyota corolla
    391                          honda civic
    392                   honda civic (auto)
    393                        datsun 310 gx
    394                buick century limited
    395    oldsmobile cutlass ciera (diesel)
    396           chrysler lebaron medallion
    398                     toyota celica gt
    399                    dodge charger 2.2
    400                     chevrolet camaro
    401                      ford mustang gl
    402                            vw pickup
    403                        dodge rampage
    404                          ford ranger
    405                           chevy s-10
    Name: name, Length: 186, dtype: object




```python
# 3 - Imprimindo carro com maior eficiência
dataset['name'][dataset['mpg'] == max(dataset['mpg'])]
```




    329    mazda glc
    Name: name, dtype: object




```python
# 4 - Imprimindo quantidade de carros fabricados depois de 77
dataset['modelyear'][dataset['modelyear'] > 77].count()
```




    150




```python
# 5 - Imprimindo cilindrada média dos carros fabricados entre 80 e 82
maior_80 = dataset[['cylinders','modelyear']][dataset['modelyear']>=80]
entre_80_82 = maior_80[['cylinders','modelyear']][maior_80['modelyear']<=82]
entre_80_82['cylinders'].mean()
```




    4.329411764705882




```python
# 6.1 - Imprimindo quantidade de carros da marca chevrolet
len(dataset[dataset['name'].str.contains('chevrolet')])
```




    43




```python
# 6.2 - Imprimindo quantidade todas as características dos carros de marca chevrolet
dataset[dataset['name'].str.contains('chevrolet')]
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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>modelyear</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504.0</td>
      <td>12.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14.0</td>
      <td>8.0</td>
      <td>454.0</td>
      <td>220.0</td>
      <td>4354.0</td>
      <td>9.0</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>chevrolet impala</td>
    </tr>
    <tr>
      <th>18</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>400.0</td>
      <td>150.0</td>
      <td>3761.0</td>
      <td>9.5</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>chevrolet monte carlo</td>
    </tr>
    <tr>
      <th>36</th>
      <td>28.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>90.0</td>
      <td>2264.0</td>
      <td>15.5</td>
      <td>71.0</td>
      <td>1.0</td>
      <td>chevrolet vega 2300</td>
    </tr>
    <tr>
      <th>42</th>
      <td>17.0</td>
      <td>6.0</td>
      <td>250.0</td>
      <td>100.0</td>
      <td>3329.0</td>
      <td>15.5</td>
      <td>71.0</td>
      <td>1.0</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>45</th>
      <td>14.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>4209.0</td>
      <td>12.0</td>
      <td>71.0</td>
      <td>1.0</td>
      <td>chevrolet impala</td>
    </tr>
    <tr>
      <th>53</th>
      <td>22.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>72.0</td>
      <td>2408.0</td>
      <td>19.0</td>
      <td>71.0</td>
      <td>1.0</td>
      <td>chevrolet vega (sw)</td>
    </tr>
    <tr>
      <th>67</th>
      <td>20.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>90.0</td>
      <td>2408.0</td>
      <td>19.5</td>
      <td>72.0</td>
      <td>1.0</td>
      <td>chevrolet vega</td>
    </tr>
    <tr>
      <th>69</th>
      <td>13.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>4274.0</td>
      <td>12.0</td>
      <td>72.0</td>
      <td>1.0</td>
      <td>chevrolet impala</td>
    </tr>
    <tr>
      <th>80</th>
      <td>13.0</td>
      <td>8.0</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>4098.0</td>
      <td>14.0</td>
      <td>72.0</td>
      <td>1.0</td>
      <td>chevrolet chevelle concours (sw)</td>
    </tr>
    <tr>
      <th>94</th>
      <td>13.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>145.0</td>
      <td>3988.0</td>
      <td>13.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>chevrolet malibu</td>
    </tr>
    <tr>
      <th>98</th>
      <td>13.0</td>
      <td>8.0</td>
      <td>400.0</td>
      <td>150.0</td>
      <td>4464.0</td>
      <td>12.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>chevrolet caprice classic</td>
    </tr>
    <tr>
      <th>105</th>
      <td>16.0</td>
      <td>6.0</td>
      <td>250.0</td>
      <td>100.0</td>
      <td>3278.0</td>
      <td>18.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>chevrolet nova custom</td>
    </tr>
    <tr>
      <th>110</th>
      <td>11.0</td>
      <td>8.0</td>
      <td>400.0</td>
      <td>150.0</td>
      <td>4997.0</td>
      <td>14.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>chevrolet impala</td>
    </tr>
    <tr>
      <th>116</th>
      <td>21.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>72.0</td>
      <td>2401.0</td>
      <td>19.5</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>chevrolet vega</td>
    </tr>
    <tr>
      <th>122</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>145.0</td>
      <td>4082.0</td>
      <td>13.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>chevrolet monte carlo s</td>
    </tr>
    <tr>
      <th>135</th>
      <td>15.0</td>
      <td>6.0</td>
      <td>250.0</td>
      <td>100.0</td>
      <td>3336.0</td>
      <td>17.0</td>
      <td>74.0</td>
      <td>1.0</td>
      <td>chevrolet nova</td>
    </tr>
    <tr>
      <th>139</th>
      <td>25.0</td>
      <td>4.0</td>
      <td>140.0</td>
      <td>75.0</td>
      <td>2542.0</td>
      <td>17.0</td>
      <td>74.0</td>
      <td>1.0</td>
      <td>chevrolet vega</td>
    </tr>
    <tr>
      <th>140</th>
      <td>16.0</td>
      <td>6.0</td>
      <td>250.0</td>
      <td>100.0</td>
      <td>3781.0</td>
      <td>17.0</td>
      <td>74.0</td>
      <td>1.0</td>
      <td>chevrolet chevelle malibu classic</td>
    </tr>
    <tr>
      <th>160</th>
      <td>18.0</td>
      <td>6.0</td>
      <td>250.0</td>
      <td>105.0</td>
      <td>3459.0</td>
      <td>16.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>chevrolet nova</td>
    </tr>
    <tr>
      <th>164</th>
      <td>15.0</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>145.0</td>
      <td>4440.0</td>
      <td>14.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>chevrolet bel air</td>
    </tr>
    <tr>
      <th>172</th>
      <td>20.0</td>
      <td>8.0</td>
      <td>262.0</td>
      <td>110.0</td>
      <td>3221.0</td>
      <td>13.5</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>chevrolet monza 2+2</td>
    </tr>
    <tr>
      <th>194</th>
      <td>17.5</td>
      <td>8.0</td>
      <td>305.0</td>
      <td>140.0</td>
      <td>4215.0</td>
      <td>13.0</td>
      <td>76.0</td>
      <td>1.0</td>
      <td>chevrolet chevelle malibu classic</td>
    </tr>
    <tr>
      <th>199</th>
      <td>22.0</td>
      <td>6.0</td>
      <td>250.0</td>
      <td>105.0</td>
      <td>3353.0</td>
      <td>14.5</td>
      <td>76.0</td>
      <td>1.0</td>
      <td>chevrolet nova</td>
    </tr>
    <tr>
      <th>202</th>
      <td>29.0</td>
      <td>4.0</td>
      <td>85.0</td>
      <td>52.0</td>
      <td>2035.0</td>
      <td>22.2</td>
      <td>76.0</td>
      <td>1.0</td>
      <td>chevrolet chevette</td>
    </tr>
    <tr>
      <th>203</th>
      <td>24.5</td>
      <td>4.0</td>
      <td>98.0</td>
      <td>60.0</td>
      <td>2164.0</td>
      <td>22.1</td>
      <td>76.0</td>
      <td>1.0</td>
      <td>chevrolet woody</td>
    </tr>
    <tr>
      <th>228</th>
      <td>17.5</td>
      <td>8.0</td>
      <td>305.0</td>
      <td>145.0</td>
      <td>3880.0</td>
      <td>12.5</td>
      <td>77.0</td>
      <td>1.0</td>
      <td>chevrolet caprice classic</td>
    </tr>
    <tr>
      <th>232</th>
      <td>17.5</td>
      <td>6.0</td>
      <td>250.0</td>
      <td>110.0</td>
      <td>3520.0</td>
      <td>16.4</td>
      <td>77.0</td>
      <td>1.0</td>
      <td>chevrolet concours</td>
    </tr>
    <tr>
      <th>237</th>
      <td>15.5</td>
      <td>8.0</td>
      <td>350.0</td>
      <td>170.0</td>
      <td>4165.0</td>
      <td>11.4</td>
      <td>77.0</td>
      <td>1.0</td>
      <td>chevrolet monte carlo landau</td>
    </tr>
    <tr>
      <th>244</th>
      <td>30.5</td>
      <td>4.0</td>
      <td>98.0</td>
      <td>63.0</td>
      <td>2051.0</td>
      <td>17.0</td>
      <td>77.0</td>
      <td>1.0</td>
      <td>chevrolet chevette</td>
    </tr>
    <tr>
      <th>260</th>
      <td>20.5</td>
      <td>6.0</td>
      <td>200.0</td>
      <td>95.0</td>
      <td>3155.0</td>
      <td>18.2</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>chevrolet malibu</td>
    </tr>
    <tr>
      <th>269</th>
      <td>19.2</td>
      <td>8.0</td>
      <td>305.0</td>
      <td>145.0</td>
      <td>3425.0</td>
      <td>13.2</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>chevrolet monte carlo landau</td>
    </tr>
    <tr>
      <th>273</th>
      <td>30.0</td>
      <td>4.0</td>
      <td>98.0</td>
      <td>68.0</td>
      <td>2155.0</td>
      <td>16.5</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>chevrolet chevette</td>
    </tr>
    <tr>
      <th>292</th>
      <td>17.0</td>
      <td>8.0</td>
      <td>305.0</td>
      <td>130.0</td>
      <td>3840.0</td>
      <td>15.4</td>
      <td>79.0</td>
      <td>1.0</td>
      <td>chevrolet caprice classic</td>
    </tr>
    <tr>
      <th>298</th>
      <td>19.2</td>
      <td>8.0</td>
      <td>267.0</td>
      <td>125.0</td>
      <td>3605.0</td>
      <td>15.0</td>
      <td>79.0</td>
      <td>1.0</td>
      <td>chevrolet malibu classic (sw)</td>
    </tr>
    <tr>
      <th>313</th>
      <td>28.8</td>
      <td>6.0</td>
      <td>173.0</td>
      <td>115.0</td>
      <td>2595.0</td>
      <td>11.3</td>
      <td>79.0</td>
      <td>1.0</td>
      <td>chevrolet citation</td>
    </tr>
    <tr>
      <th>318</th>
      <td>32.1</td>
      <td>4.0</td>
      <td>98.0</td>
      <td>70.0</td>
      <td>2120.0</td>
      <td>15.5</td>
      <td>80.0</td>
      <td>1.0</td>
      <td>chevrolet chevette</td>
    </tr>
    <tr>
      <th>320</th>
      <td>28.0</td>
      <td>4.0</td>
      <td>151.0</td>
      <td>90.0</td>
      <td>2678.0</td>
      <td>16.5</td>
      <td>80.0</td>
      <td>1.0</td>
      <td>chevrolet citation</td>
    </tr>
    <tr>
      <th>348</th>
      <td>23.5</td>
      <td>6.0</td>
      <td>173.0</td>
      <td>110.0</td>
      <td>2725.0</td>
      <td>12.6</td>
      <td>81.0</td>
      <td>1.0</td>
      <td>chevrolet citation</td>
    </tr>
    <tr>
      <th>375</th>
      <td>28.0</td>
      <td>4.0</td>
      <td>112.0</td>
      <td>88.0</td>
      <td>2605.0</td>
      <td>19.6</td>
      <td>82.0</td>
      <td>1.0</td>
      <td>chevrolet cavalier</td>
    </tr>
    <tr>
      <th>376</th>
      <td>27.0</td>
      <td>4.0</td>
      <td>112.0</td>
      <td>88.0</td>
      <td>2640.0</td>
      <td>18.6</td>
      <td>82.0</td>
      <td>1.0</td>
      <td>chevrolet cavalier wagon</td>
    </tr>
    <tr>
      <th>377</th>
      <td>34.0</td>
      <td>4.0</td>
      <td>112.0</td>
      <td>88.0</td>
      <td>2395.0</td>
      <td>18.0</td>
      <td>82.0</td>
      <td>1.0</td>
      <td>chevrolet cavalier 2-door</td>
    </tr>
    <tr>
      <th>400</th>
      <td>27.0</td>
      <td>4.0</td>
      <td>151.0</td>
      <td>90.0</td>
      <td>2950.0</td>
      <td>17.3</td>
      <td>82.0</td>
      <td>1.0</td>
      <td>chevrolet camaro</td>
    </tr>
  </tbody>
</table>
</div>



## Tarefa 4: Estatística Descritiva

Para o atributo alvo 'mpg', calcule:
 1. Média
 2. Mediana
 3. Máximo
 4. Mínimo
 5. Desvio Padrão
 6. Skewness
 7. Curtose
 6. Há outliers? (Valores de mpg acima ou abaixo da média + 2 desvios padrões)?
 7. Responda: O que se pode afirmar a respeito da distribuição de dados desse atributo?


```python
# 1 - Média
dataset['mpg'].mean()
```




    23.44591836734694




```python
# 2 - Mediana
dataset['mpg'].median()
```




    22.75




```python
# 3 - Máximo
dataset['mpg'].max()
```




    46.6




```python
# 4 - Mínimo
dataset['mpg'].min()
```




    9.0




```python
# 5 - Desvio padrão
dataset['mpg'].std()
```




    7.805007486571799




```python
# 6 - Skewness
dataset['mpg'].skew()
```




    0.45709232306041025




```python
# 7 - Curtose
dataset['mpg'].kurtosis()
```




    -0.5159934946351457




```python
# 8 - Há outliers neste dataset?
media = dataset['mpg'].mean()
dp = dataset['mpg'].std()
dataset['mpg'][dataset['mpg'] > media+2*dp].count()>0 or dataset['mpg'][dataset['mpg'] < media-2*dp].count()>0
```




    True



####  9  - Responda: O que se pode afirmar a respeito da distribuição de dados desse atributo?
##### Resposta:
Se observarmos a moda da distribuição:


```python
dataset['mpg'].mode()
```




    0    13.0
    dtype: float64



Temos que:

    Média = 23.44591836734694
    Mediana = 22.75
    Moda = 13
    
ou seja:

    Moda<Mediana<Média
    
Portanto, trata-se de uma distribuição assimétrica positiva. Os dados se encontram, principalmente, no início da distribuição.

## Tarefa 5: Identificando correlações

1. Qual a correlação entre o peso do chassi (weight) e o consumo (mpg)?
2. Essa medida de correlação é mais forte ao considerarmos apenas os carros da marca Toyota?
3. Qual a correlação entre a potência (horsepower) e o consumo (mpg) para os veículos do dataset?


```python
# 1 - Imprimindo correlação entre o peso do chassi e consumo
correlacoes = dataset.corr()
correlacoes['weight']['mpg']
```




    -0.8322442148315756




```python
# 2 - Imprimindo se a medida de correlação é mais forte ao considerarmos apenas os carros da marca Toyota
correlacoes_toyota = dataset[dataset['name'].str.contains('toyota')].corr()
abs(correlacoes_toyota['weight']['mpg']) > abs(correlacoes['weight']['mpg'])
```




    False




```python
# 3 - Imprimindo a correlação entre potência e consumo
correlacoes['horsepower']['mpg']
```




    -0.7784267838977751



## Tarefa 6: Preparando uma partição do tipo Holdout

1. Remova do dataset a coluna 'name'
2. Exclua o atributo alvo mpg do dataset e o atribua a uma variável Y
3. Efetue uma partição holdout 70/30 utilizando o sci-kit learn.
 - Para auxiliar, consulte a [documentação](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)


```python
# 1 - Removendo do dataset a coluna 'name'
dataset.drop(['name'], axis=1, inplace=True)
```


```python
# 2.1 - Atribuindo o atributo mpg a Y
y = dataset['mpg']

# 2.2 - Excluindo atributo mpg
dataset.drop(['mpg'], axis=1, inplace=True)
```


```python
# 3 - Partição holdout 70/30
X_train, X_test, Y_train, Y_test = train_test_split(dataset, y, test_size=0.30)
```


```python
392*0.30
```




    117.6




```python
len(X_test)
```




    118


