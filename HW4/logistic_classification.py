import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()

def get_data(n1,n2,fd):
    '''
    Get the data from txt file 
    '''
    features = np.genfromtxt(fd)
    num_id = [n1,n2]
    data = np.zeros(3)

    for i in range(features.shape[0]):
        if int(features[i][0]) in num_id:
            if features[i][0] == num_id[0]:
              data = np.vstack((data, np.array([features[i][1],features[i][2],1.]) ))
            else:
              data = np.vstack((data, np.array([features[i][1],features[i][2],0.]) )) 

    
    df = pd.DataFrame(data=data[1:,:],columns=["Symmetry","Intensity","is_{}".format(n1)])
    df['is_{}'.format(n1)] = df['is_{}'.format(n1)].astype(int)

    return data[1:,:],df

class LogisticRegression:
    def __init__(self, lr=0.1, num_iter=100000, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros((X.shape[1]))
        self.iter_vals = []
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size

            self.theta -= self.lr * gradient
            
            if(i % 1000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                self.iter_vals.append(self.__loss(h,y))
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return (self.predict_prob(X) >= threshold) * 1

    def get_boundary(self,X):
        plot_x = np.array([min(X[:,0]), max(X[:,0])])
        plot_y = (-1/self.theta[2]) * (self.theta[1] * plot_x + self.theta[0])
        
        return plot_x,plot_y


class LinearRegression:
    def __init__(self, lr=0.0001, num_iter=100000, fit_intercept=True):
        self.lr = lr 
        self.num_iter = num_iter 
        self.fit_intercept = fit_intercept

    def __add_intercept(self, A):
        intercept = np.ones((A.shape[0], 1))
        return np.concatenate((intercept, A), axis=1)

    def __objfunc(self,x):
        b_ = np.dot(self.A,x)
        norm = np.linalg.norm( (b_ - self.b) )**2

        return .5 * norm

    def __gradient(self,x):
        tmp1 = np.dot( np.dot(self.A.T,self.A), x)
        tmp2 = np.dot(self.A.T,self.b)

        return tmp1 - tmp2

    def fit(self,A,b):
        if self.fit_intercept:
            self.A = self.__add_intercept(A)
        else:
            self.A = A 
        
        self.b = b.reshape(-1,1)
        self.theta = np.zeros((self.A.shape[1],1))
        
        self.iter_vals = []
        for i in range(self.num_iter):

            self.theta -= self.lr * self.__gradient(self.theta)

            if (i % 1000 == 0):
                z = self.__objfunc(self.theta)
                self.iter_vals.append(z)

    def predict(self,A):
        if self.fit_intercept:
            A = self.__add_intercept(A)
        pred = np.sign( np.dot(A,self.theta) )
        return pred
        
    def get_boundary(self,A):
        plot_x = np.array([min(A[:,0]), max(A[:,0])])
        plot_y = (-1/self.theta[2]) * (self.theta[1] * plot_x + self.theta[0])
        
        return plot_x,plot_y
            


# Choose to try to classify digit 1 from 9
n1,n2 = 1,9
data,df = get_data(n1,n2,"features_train.txt")
data2,df2 = get_data(n1,n2,"features_test.txt")
X_train,y_train = data[:,:-1],data[:,-1]
X_test,y_test = data2[:,:-1],data2[:,-1]

############################################### Logistic Regression

m = LogisticRegression()
m.fit(X_train,y_train)

plt.figure()
plot_x, plot_y = m.get_boundary(X_train)
ax = sns.scatterplot(x="Symmetry",y="Intensity",hue="is_{}".format(n1),data=df)
plt.plot(plot_x,plot_y,color='r')
plt.title("Decision Boundary Plot for '1 or 9' Digit Classification")
plt.show()

plt.figure()
plt.plot(range(len(m.iter_vals)),m.iter_vals)
plt.title("Plot of Loss Function over Iterations (lr = 0.0001)")
plt.xlabel("Iteration Number x1000 (iter = x * 1000)")
plt.ylabel("Objective Function Value")
plt.xlim([0,20])
plt.show()

# Get accuracy on test data
predictions = m.predict(X_test,.5)
acc = ((predictions == y_test)*1).sum() / len(predictions)
print("")
print("Accuracy of Logistic Model [TEST] = {:.2f}%".format(acc*100))

plt.figure()
m.fit(X_test,y_test)
plot_x, plot_y = m.get_boundary(X_test)
ax = sns.scatterplot(x="Symmetry",y="Intensity",hue="is_{}".format(n1),data=df2)
plt.plot(plot_x,plot_y,color='r')
plt.title("Decision Boundary Plot for '1 or 9' Digit Classification (Test Data)")
plt.show()

##############################################


############################################### Linear Regression

lin_m = LinearRegression()

y_test = y_test * 2 - 1
y_train = y_train * 2 - 1
lin_m.fit(X_train,y_train)

plt.figure()
plot_x, plot_y = lin_m.get_boundary(X_train)
ax = sns.scatterplot(x="Symmetry",y="Intensity",hue="is_{}".format(n1),data=df)
plt.plot(plot_x,plot_y,color='r')
plt.title("Decision Boundary Plot for '1 or 9' Digit Classification (Train Data)")
plt.show()

plt.figure()
plt.plot(range(len(lin_m.iter_vals)),lin_m.iter_vals)
plt.title("Plot of Loss Function over Iterations (lr = 0.0001)")
plt.xlabel("Iteration Number x1000 (iter = x * 1000)")
plt.ylabel("Objective Function Value")
plt.xlim([0,20])
plt.show()

# Get accuracy on test data
predictions_lin = lin_m.predict(X_test)
acc = ((predictions_lin.reshape(1,-1) == y_test)*1).sum() / len(predictions_lin)
print("")
print("Accuracy of Linear Model [TEST] = {:.2f}%".format(acc*100))

plt.figure()
lin_m.fit(X_test,y_test)
plot_x, plot_y = lin_m.get_boundary(X_test)
ax = sns.scatterplot(x="Symmetry",y="Intensity",hue="is_{}".format(n1),data=df2)
plt.plot(plot_x,plot_y,color='r')
plt.title("Decision Boundary Plot for '1 or 9' Digit Classification (Test Data)")
plt.show()


##############################################


