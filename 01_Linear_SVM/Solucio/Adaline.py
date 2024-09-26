import numpy as np

class Adaline:
    """ADAptive LInear NEuron classifier.
       Gradient Descent

    Parametres
    ------------
    eta : float
        Rati d'aprenentatge (Learning Rate) [0.0, 1.0]
    n_iter : int
       Nombre d'iteracions sobre el conjunt d'entrenament.

    Attributes
    -----------
    w_ :  llista (array, no canvia de mida)
        Pesos després de l'entrenament.
    errors_ : llista
        Error in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Entrenament.

        Parameters
        ----------
        X: {array}, shape = [n_mostres, n_features]
            Vectors d'entrenament, on n_mostres és el nombre de mostres i n_features és el nombre de funcions.
        y: array, shape = [n_mostres]
            Etiquetes.
        Returns
        -------
        self
        """

        self.w_ = np.zeros(1 + X.shape[1])  # pesos de la xarxa
        self.cost_ = [] # valor de la funció de cost a cada iteració

        for i in range(self.n_iter):
            output = self.net_input(X) # Obtenim la sortida de la xarxa
            errors = (y - output) # calculam l'error entre la sortida i l'etiqueta
            self.w_[1:] += self.eta * X.T.dot(errors) # actualitzam els pesos
            self.w_[0]  += self.eta * errors.sum() # actualitzam el bias

            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Sortida de la xarxa sense aplicar la funció escalo"""
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def predict(self, X):
        """Predicció de la xarxa, un cop hem binaritzat la seva sortida"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
