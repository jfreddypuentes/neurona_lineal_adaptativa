from matplotlib.colors import ListedColormap
from AdalineGD import AdalineGD
import matplotlib.pyplot as plt
import numpy as np
import iris_data as iris

# Obtencion de datos
X, y = iris.get_train_target_data()

#Normalizando los datos. (media 0, std 1)
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

#red1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
red1 = AdalineGD(n_iter=10, eta=0.01).fit(X_std, y)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

# Gráfico Izquierda: El error es mayor en cada epoca. No minimiza la funcion de coste.
ax[0].plot(range(1, len(red1.cost_) + 1), np.log10(red1.cost_), marker='o')
ax[0].set_xlabel('Epocas')
ax[0].set_ylabel('log(suma error cuadratico)')
ax[0].set_title('Adaline - Tasa de aprendizaje 0.01')


red2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)

# Gráfico Derecha: El coste disminuye pero dado que la tasa de aprendizaje es bajo
#                  entonces, requerirá demasiadas epocas (ciclos) para converger  al
#                  minimo coste global.
ax[1].plot(range(1, len(red2.cost_) + 1), np.log10(red2.cost_), marker='o')
ax[1].set_xlabel('Epocas')
ax[1].set_ylabel('suma error cuadratico')
ax[1].set_title('Adaline - Tasa de aprendizaje 0.0001')


plt.show()
