import numpy as np

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        ''' Calculo y actualizacion de pesos '''
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        ''' Calcula la entrada de la red. (entrada * pesos) + sesgo '''
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        ''' Calcula la activacion lineal '''
        return X

    def predict(self, X):
        ''' Retorna la clase clasificada. 1 si >=0, -1 si <0 '''
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
