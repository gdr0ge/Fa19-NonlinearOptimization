import numpy as np 
import numpy.linalg

def polyKernel(a,b,pwr):
    return np.dot(a,b)**pwr #np.dot(a,a) - np.dot(b,b) # -1 #

def rbfKernel(a,b,gamma):
    return np.exp(-gamma * np.linalg.norm(a - b))

class DualSVM:
    
    def __init__(self, C, tolerance = 1, kernel = np.dot, kargs = () ):

        self.C = C
        self.kernel = kernel
        self.tolerance = tolerance
        self.kargs = kargs
        self.verbose = True


    def fit(self, X, y):
        
        ''' Construct the Q matrix for solving '''     
        Q = np.zeros((len(X),len(X)))
        for i in range(len(X)):
            for j in range(i,len(X)):
                Qval = y[i] * y[j]
                Qval *= self.kernel(*(
                                (X[i], X[j])
                                + self.kargs
                                ))
                Q[i,j] = Q[j,i] = Qval
        
        ''' Solve for a and w (x in pdf theory) simultaneously by coordinate descent '''
        self.w = np.zeros(X.shape[1])
        self.a = np.zeros(X.shape[0])
        delta = 10000000000.0
        while delta > self.tolerance:
            delta = 0.
            for i in range(len(X)):
                g = np.dot(Q[i], self.a) - 1.0
                adelta = self.a[i] - min(max(self.a[i] - g/Q[i,i], 0.0), self.C) 
                self.w += adelta * X[i]
                delta += abs(adelta)
                self.a[i] -= adelta

            # if self.verbose:
            #     print("Descent step magnitude:", delta)

        self.sv = X[self.a > 0.0, :]
        self.a = (self.a * y)[self.a > 0.0]
        
        ''' Solve for b '''
        self.b = self._predict(self.sv[0,:])[0]
        if self.a[0] > 0:
            self.b *= -1
        
    
    def _predict(self, X):
        if (len(X.shape) < 2):
            X = X.reshape((1,-1))
        clss = np.zeros(len(X))
        for i in range(len(X)):
            for j in range(len(self.sv)):
                clss[i] += self.a[j] * self.kernel(* ((self.sv[j],X[i]) + self.kargs))
        return clss
    
    def predict(self, X):
        ''' Get predictions '''
        pred = self._predict(X) > self.b 
        return np.where(pred == True, 1, -1)

    def get_boundary(self, x):
        plot_x = np.array([min(x[:,0]), max(x[:,0])])
        plot_y = plot_x * (self.w[1] / self.w[0] + self.b)
        
        return plot_x,plot_y


