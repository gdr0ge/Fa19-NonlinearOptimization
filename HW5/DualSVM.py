import numpy as np 
import numpy.linalg


def get_data(n1,n2,fd):
    '''
    Get the data from txt file 
    '''
    data = pd.read_csv(fd, sep=" ", header=None)
    data = data[:257]

    num_id = [n1,n2]
    data_t = np.zeros(257)

    for i in range(test_features.shape[0]):
        if int(test_features[i][0]) in num_id:
            feats = test_features[i][1:]
            if test_features[i][0] == num_id[0]:
                tmp = np.concatenate((feats, np.array([1])),axis=None)
                data_t = np.vstack((data_t, tmp))
            else:
                tmp = np.concatenate((feats, np.array([0])), axis=None)
                data_t = np.vstack((data_t, tmp)) 

    data_t = data_t[1:]
    df = pd.DataFrame(data=data_t)

    return data_t, df


def rbfKernel(a,b,gamma):
    return np.exp(-gamma * np.linalg.norm(a - b))

class DualSVM:
    
    def __init__(self, C=1, epsilon = .001, kernel = np.dot, kargs = () ):

        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.kargs = kargs
        self.verbose = True


    def fit(self, X, y):
        '''
            X : data matrix
            y : labels as [1,-1]
        '''

        ''' Construct the Q matrix for solving '''     
        Q = np.zeros((len(X),len(X)))
        for i in range(len(X)):
            for j in range(i,len(X)):
                Qval = y[i] * y[j]
                Qval *= self.kernel(*((X[i], X[j]) + self.kargs))
                Q[i,j] = Q[j,i] = Qval
        
        ''' Solve for a and w (x in pdf theory) simultaneously by coordinate descent '''
        self.w = np.zeros(X.shape[1])
        self.a = np.zeros(X.shape[0])
        delta = 100000
        while delta > self.epsilon:
            delta = 0.
            for i in range(len(X)):
                tmp = np.dot(Q[i], self.a) - 1.0
                proj = self.a[i] - min(max(self.a[i] - tmp/Q[i,i], 0.0), self.C) 
                self.w += proj * X[i]
                delta += abs(proj)
                self.a[i] -= proj

        self.sv = X[self.a > 0.0]
        self.a = (self.a * y)[self.a > 0.0]
        
        ''' Solve for b '''
        self.b = self._predict(self.sv[0])[0]
        if self.a[0] > 0:
            self.b *= -1
        
    
    def _predict(self, X):
        if (len(X.shape) < 2):
            X = X.reshape((1,-1))
        pred = np.zeros(len(X))
        for i in range(len(X)):
            for j in range(len(self.sv)):
                pred[i] += self.a[j] * self.kernel(* ((self.sv[j],X[i]) + self.kargs))
        return pred
    
    def predict(self, X):
        ''' Get predictions '''
        pred = self._predict(X) > self.b 
        return np.where(pred == True, 1, -1)

    def get_boundary(self, x):
        plot_x = np.array([min(x[:,0]), max(x[:,0])])
        plot_y = plot_x * (self.w[1] / self.w[0] + self.b)
        
        return plot_x,plot_y


