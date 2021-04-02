import numpy as np 

B = .5
sig = .1
s = 1

xk = np.array([1,-2])[:,None]
m = 0

hess = np.array([[6,0],[0,48]])

def fn(xk):
    f = 3*xk.ravel()[0]**2 + xk.ravel()[1]**4
    return f

def grad_fn(xk):
    f = np.array([6*xk.ravel()[0],4*xk.ravel()[1]**3])[:,None]
    return f

def hessian(xk):
    f = np.array([[6,0],[0,12*xk[1]**2]])
    return f

def check_con(xk,m):
    dk = (-1)* np.dot( np.linalg.inv(hessian(xk)) , grad_fn(xk) )
    fk = fn(xk)

    fk_step = fn( xk + B**m * s * dk )
    return (fk - fk_step) >= (-sig * B**m * s * np.dot(grad_fn(xk).T , dk)) 

while (not check_con(xk,m).all()):
    m += 1

print(m)