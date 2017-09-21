# Numeros - Sistemas Inteligentes
Sistema de reconhecimento de números feito com python, utilizando redes neurais.

Nossa rede neural trabalhará com 25 entradas, 15 camadas ocultas e 10 saídas.

Também usaremos o Bias ativo.

# Os números
Os números serão como a imagem abaixo, composto por uma matriz 5 x 5, onde o valor de cada pixel servirá para que a rede neural seja treinada e reconheça o número.
Os Pixels assumirão valores 0 (Para os pixels brancos) e 1 (Para os pretos), formando um vetor que representará o número.

Abaixo, a imagem contento todos os números.

![Imagem de todos os números](https://github.com/wallacehenriquesilva/numeros-si/blob/master/Numeros.PNG)

Abaixo, um exemplo de número e seu respectivo vetor.

![Imagem do número exemplo](https://github.com/wallacehenriquesilva/numeros-si/blob/master/numero0.PNG)

[0] = [0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0]


# O Resultado

O resultado, será representado por um vetor com valor 1 na posição no número correspondente.


Ex. 

![Imagem do número exemplo](https://github.com/wallacehenriquesilva/numeros-si/blob/master/numero8.PNG)
8 = [0,0,0,0,0,0,0,0,1,0]