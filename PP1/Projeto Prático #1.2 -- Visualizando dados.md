
# Projeto Prático 1.2 -- Explorando Dados

O projeto prático 1.2 da disciplina de Redes Neurais Artificiais tem como ideia geral seguir o passo a passo das atividades solicitadas para aprender a utilizar as bibliotecas Python para praticar os conceitos de visualização de dados discutidos ao longo dessas primeiras aulas.

Na avaliação será levado em conta:
1. Corretude das tarefas solicitadas
2. Qualidade e boas práticas de codificação
3. Qualidade dos gráficos


Preecha aqui os integrantes da dupla e suas respectivas matrículas (duplo clique para editar):
- Jean Phelipe de Oliveira Lima - 1615080096

## Apresentação da Atividade

Continuaremos utilizando o dataset 'autompg.csv' e vamos concentrar o uso na biblioteca matplotlib.pyplot, cuja documentação pode ser encontrada aqui: https://matplotlib.org/


```python
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# Abrindo o dataset
dataset = pd.read_csv('autompg.csv', encoding='UTF-8', sep=';')
# Apagando exemplos com dados faltosos
dataset.dropna(inplace=True)
```

## Atividade 7 -- Histograma

Construa o histograma do atributo alvo mpg utilizando bins de tamanho 1. O seu histograma deve ser intitulado 'Consumo', o eixo 'y' deve apresentar a unidade de medida 'milhas por galão'. O eixo x deve ser rotulado com os valores associados aos bins. A cor do gráfico deve ser em tons de cinza.


```python
aux=[]
for i in dataset['mpg']:
    if i not in aux:
        aux.append(i)

plt.hist(dataset['mpg'], bins=len(aux),color='#808080', orientation='horizontal')
plt.title('Consumo')
plt.ylabel('milhas por galão')
```




    Text(0,0.5,'milhas por galão')




<img src="https://raw.githubusercontent.com/jpdol/RedesNeurais/master/PP1/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_files/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_4_1.png">


## Atividade 8 -- Boxplot

1. Obtenha o boxplot do atributo alvo mpg
2. Obtenha o boxplot do peso dos carros
3. Responda: o peso dos carros distribuído de maneira simétrica?


```python
# 1 - Boxplot - mpg
plt.boxplot(dataset['mpg'])
plt.title('Boxplot - Milhas por Galão')
```




    Text(0.5,1,'Boxplot - Milhas por Galão')




<img src="https://raw.githubusercontent.com/jpdol/RedesNeurais/master/PP1/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_files/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_6_1.png">



```python
# 2 - Boxplot - peso do carro
plt.boxplot(dataset['weight'])
plt.title('Boxplot - Peso do Carro')
```




    Text(0.5,1,'Boxplot - Peso do Carro')




<img src="https://raw.githubusercontent.com/jpdol/RedesNeurais/master/PP1/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_files/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_7_1.png">


#### O peso dos carros é distribuído de maneira simétrica?
##### Resposta:
Não, pois a maior parte dos valores concentram-se na parte inferior do gráfico

## Atividade 10 -- Scatter Plot

Produza um scatterplot dos atributos 'weight' e 'mpg'


```python
plt.scatter(dataset['mpg'], dataset['weight'])
plt.xlabel('Milhas por galão')
plt.ylabel('Peso do Carro')
plt.title('Scatterplot - peso x mpg')
```




    Text(0.5,1,'Scatterplot - peso x mpg')




<img src="https://raw.githubusercontent.com/jpdol/RedesNeurais/master/PP1/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_files/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_10_1.png">


## Atividade 11 -- Boxplot categorizado

1. Construa boxplots do atributo 'mpg' categorizados por ano, isto é, uma única figura contendo vários boxplots do consumo dos veículos agrupados segundo o ano.
2. É possível observar uma tendência de aumento de eficiência (maior mpg) com o passar dos anos?


```python
# 1 - Imprimindo boxplot de mpg categorizado por ano
anos = []
for i in dataset['modelyear']:
    if i not in anos:
        anos.append(int(i))
aux=[]
for i in range(len(anos)):
    aux.append(i+1) 
    
aux2=[]
for i in anos:
    aux2.append(dataset['mpg'][dataset['modelyear']==i])
    
plt.boxplot(aux2)
plt.title('Boxplot - mpg por ano')
plt.xticks(aux, anos)
plt.xlabel('Anos')
plt.ylabel('milhas por galão')
```




    Text(0,0.5,'milhas por galão')




<img src="https://raw.githubusercontent.com/jpdol/RedesNeurais/master/PP1/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_files/Projeto%20Pr%C3%A1tico%20%231.2%20--%20Visualizando%20dados_12_1.png">


#### 2 - É possível observar uma tendência de aumento de eficiência (maior mpg) com o passar dos anos?
##### Resposta:
Sim, não é um crescimento uniforme, no entento é possível perceber que há o crescimento em eficiência ao observarmos os valores de mpg para as regiões interquartis das categorias apresentadas.
