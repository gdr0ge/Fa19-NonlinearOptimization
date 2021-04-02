import numpy as np  
import time
import matplotlib.pyplot as plt

# Code to load in the Q matrix and b vector
Q = np.genfromtxt('Q.csv',delimiter=',')
b = np.genfromtxt('b.csv', delimiter=',')[:,None]

# Our initial starting point for all algorithms
x0 = np.ones(b.shape)

# Parameters for Armijo step size rule
B = .25
sig = .1
s = 1

# This will be our closed form solution that we will 
# check our approximations againsnt 
closed_form_solution = np.dot( np.linalg.inv(Q), -b )

# Function we are trying to minimize
def fn(xk):
    Qx = np.dot(Q,xk)
    bTx = np.dot(b.T,xk)

    return 1/2 * np.dot(xk.T,Qx) + bTx 

# Gradient of objective function
def grad(xk):
    term1 = 1/2 * np.dot((Q + Q.T),xk)

    return term1 + b

# Armijo step size check for computing m
def armijo_chk(xk,m):

    dk = -grad(xk)
    fk = fn(xk)

    fk_step = fn( xk + B**m * s * dk )

    return (fk - fk_step) >= (-sig * B**m * s * np.dot( grad(xk).T , dk ))

# Function that implements steepes descent with Armijo step
# size rule
def use_armijo(B,sigma,s,start_time):

    xk = x0
    m = 1
    epsilon = .0001

    # Various values to store
    values = [xk]
    diffs = [ abs(fn(closed_form_solution) - fn(xk)) ]
    times = [0]

    for i in range(1000):
        
        # Our inequality check to find next step size
        while (not armijo_chk(xk,m).item()):
            m += 1
        
        # Compute next step
        xk_1 = xk - B**m * s * grad(xk)

        # Get change in time and store iterate
        # store difference from minimal
        times.append( time.clock() - start_time )
        values.append(xk_1)
        
        diff = abs(fn(closed_form_solution) - fn(xk_1))
        diffs.append(diff)

        # Reset m
        m = 1

        # A check to see if we are close to minimum
        if abs((fn(xk_1)-fn(xk))) <= epsilon:
            break

        xk = xk_1

    return values, diffs, times


# Diagonally sclaed gradient method with constant step size
def use_dsg(s,start_time):
    # started with s = 1 but did not converge
    # using s = .5 --> converges

    xk = x0 
    epsilon = .0001

    # The Hessian diagonals that will be used 
    # for the new step direction
    Dk = np.linalg.inv( np.diag(np.diag(Q)) ) 

    # Various values to store
    values = [xk]
    diffs = [ abs(fn(closed_form_solution) - fn(xk)) ]
    times = [0]

    for i in range(1000):

        # Compute next step
        xk_1 = xk - s * np.dot(Dk,grad(xk))

        # Get change in time and store iterate
        # store difference from minimal
        times.append( time.clock() - start_time )
        values.append(xk_1)

        diff = abs(fn(closed_form_solution) - fn(xk_1))
        diffs.append(diff)

        # A check to see if we are close to minimum
        if abs((fn(xk_1)-fn(xk))) <= epsilon:
            break

        xk = xk_1

    return values, diffs, times


# Conjuage gradient method with exact minimization step size rule
def use_cgm(start_time):

    xk = x0

    # Set up the first residual vector 
    r = b + np.dot(Q, xk) 

    # Residual will be first search direction
    d = r

    d_new = np.dot( r.T , r ) 

    vals = [xk]
    diffs = [ abs(fn(closed_form_solution) - fn(xk)) ]
    times = [0]

    for i in range(len(b)):

        # Compute alpha to formulate new iterate step
        Qd = np.dot( Q, d )
        a = d_new / np.dot( d.T, Qd )

        # Compute new iterate
        xk_1 = xk + a * d

        # Store various values
        times.append( time.clock() - start_time )
        vals.append(xk_1)

        diff = abs(fn(closed_form_solution) - fn(xk_1))
        diffs.append(diff)

        # Compute new residual vector 
        r = b + np.dot( Q, xk_1 )

        # Store old d and compute new
        d_old = d_new
        d_new = np.dot( r.T, r )

        # Compute beta to determine new search direction
        beta =  d_new / d_old

        # Compute new search direction
        d = r - beta * d 

        xk = xk_1

    return vals,diffs,times

print("Closed form solution: {}".format(fn(closed_form_solution).item()),end="\n\n")

print("Starting steepest descent [Armijo]...")
t_start = time.clock()
vals,diffs_a,times1 = use_armijo(B,sig,s,t_start)
diffs_a = [x.item() for x in diffs_a]
t_stop = time.clock()

print("Armijo: time = {} converge point = {}".format((t_stop - t_start),fn(vals[-1]).item()),end="\n\n")

print("Starting diagonally scaled gradient...")

t_start = time.clock()
vals,diffs_d,times2 = use_dsg(.5,t_start)
diffs_d = [x.item() for x in diffs_d]
t_stop = time.clock()

print("DSG: time = {} converge point = {}".format((t_stop - t_start),fn(vals[-1]).item()),end="\n\n")


print("Starting Conjugate Gradient Method...")

t_start = time.clock()
vals,diffs_c,times3 = use_cgm(t_start)
diffs_c = [x.item() for x in diffs_c]
t_stop = time.clock()

print("CGM: time = {} converge point = {}".format((t_stop - t_start),fn(vals[-1]).item()),end="\n\n")


plt.figure(figsize=(12,8))

plt.subplot(1,1,1)
plt.title("Minimization Algorithm Objective Value Comparisons n = 50 c = 5000",fontsize=20)
plt.xlabel("Iterate",fontsize=18)
plt.ylabel("Min val difference for iterate xk",fontsize=18)
plt.xlim([0,15])
plt.plot(range(len(diffs_a)),diffs_a,label="arm")
plt.plot(range(len(diffs_d)),diffs_d,label="dsg")
plt.plot(range(len(diffs_c)),diffs_c,label="cgm")

plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(12,8))

plt.subplot(1,1,1)
plt.title("Minimization Algorithm Time Comparisons n = 50 c = 5000",fontsize=20)
plt.xlabel("Iterate",fontsize=18)
plt.ylabel("Elasped Time",fontsize=18)
plt.xlim([0,min([len(times1),len(times2),len(times3)])])
plt.ylim([0,.015])
# plt.ylim([0,max([times1[-1],times2[-1],times3[-1]])])
plt.plot(range(len(times1)),times1,label="arm")
plt.plot(range(len(times2)),times2,label="dsg")
plt.plot(range(len(times3)),times3,label="cgm")

plt.legend(loc='upper right')
plt.show()
