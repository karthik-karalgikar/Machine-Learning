import numpy as np

class MyPerceptron:

    def __init__(self, learning_rate=0.1, n_iterations=1000):

        self.lr = learning_rate 
        self.epochs = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        print(X.shape)
        print(y.shape)
        


        pass

    def predict(self, X):

        pass
    