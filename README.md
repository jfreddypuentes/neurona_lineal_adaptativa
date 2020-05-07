# Neurona Lineal Adaptativa
La neuroa lineal adaptativa es un tipo de red neuronal de capa única. Mejora al perceptrón agregando una función de coste con la que actualiza los pesos.
Este tipo de red sienta las bases para la clasificación, regresión logistica y máquinas de vectores de soporte.

### Función de coste
La función de coste es la función que actualiza los pesos. Es un función convexa. Por lo que se usa el potente algoritmo de optimización descenso de gradiente para encontrar el minimo local. 

### Obtención de datos
~~~
X, y = iris.get_train_target_data()
~~~

### Inicialización de la red
~~~
red1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
~~~

### Resultados con tasas de aprendizaje distintas
![Resultados](/tasa_aprendizaje_epocas.png)