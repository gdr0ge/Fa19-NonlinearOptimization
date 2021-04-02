import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


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


class SVM:
	def __init__(self, lr=0.01, C=5.0, num_iter=10000, fit_intercept=True):
		self.lr = lr 
		self.C = C
		self.num_iter = num_iter
		self.fit_intercept = fit_intercept

	def __add_intercept(self, x):
		intercept = np.ones((x.shape[0], 1))
		return np.concatenate((intercept, x), axis=1)

	def __objective(self, x, y):
		zeros = np.zeros((x.shape[0],1))
		tmp = 1 - np.dot(self.w.T,x.T).reshape(-1,1)

		return .5 * np.dot(self.w.T,self.w) + self.C * np.maximum(zeros,tmp).sum() / x.shape[0]

	def __sub_gradient(self, x, y):

		# Sub-gradient
		idx = np.random.randint(x.shape[0])
		xi, yi = x[idx].reshape(-1,1), y[idx]
		
		if max(0, 1 - (yi*np.dot(self.w.T,xi)).item() ) == 0:
			grad = self.w
		else:
			grad = self.w - self.C * yi * xi

		return grad


	def fit(self, x, y):
		'''
            X : data matrix
            y : labels as [1,-1]
        '''

		if self.fit_intercept:
			x = self.__add_intercept(x)

		self.w = np.zeros((x.shape[1],1))
		self.iter_vals = []
		for i in range(self.num_iter):

			self.w -= self.lr * self.__sub_gradient(x, y)

			if (i % 1000 == 0):
				val = self.__objective(x, y)
				self.iter_vals.append(val)

	def predict(self, x):
		if self.fit_intercept:
			x = self.__add_intercept(x)

		pred = np.sign(np.dot(x,self.w)).reshape(1,-1)
		return pred


	def get_boundary(self, x):
		plot_x = np.array([min(x[:,0]), max(x[:,0])])
		plot_y = (-1/self.w[2]) * (self.w[1] * plot_x + self.w[0])
		
		return plot_x,plot_y
		
