# import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 



def gen_data():
    '''
    Function to generate the random data
    '''
    A = np.random.random_sample((50,10)) 
    b = np.random.random_sample((50,1)) 

    A = 2 * A - .5

    return A,b


def armijo_chk(xk,m):
    '''
    Function that performs armijo inequality check
    '''
    dk = -grad_obj(xk)
    fk = obj_func(xk)

    fk_step = obj_func( xk + B**m * s * dk )

    return (fk - fk_step) >= (-sig * B**m * s * np.dot( grad_obj(xk).T , dk ))


def use_armijo(B,sigma,s):

    global A 
    global b
    global thresh

    x0 = np.ones((A.shape[1],1))
    xk = x0
    m = 1
    epsilon = 1e-4

    # Various values to store
    values = [xk]

    # Number of inequality check iterations
    m_iter = 0

    for i in range(1000):
        
        # Our inequality check to find next step size
        while (not armijo_chk(xk,m).item()):
            m += 1
        
        # Compute next step
        xk_1 = xk - B**m * s * grad_obj(xk)

        # Get change in time and store iterate
        # store difference from minimal
        values.append(xk_1)

        # Reset m
        m_iter += m
        m = 1

        # A check to see if we are close to minimum
        if abs((obj_func(xk_1)-obj_func(xk))) <= epsilon:
            break

        xk = xk_1

    return values, m_iter


def obj_func(x):
    '''
    The objective function we are trying to minimize
    '''
    global A 
    global b 

    b_ = np.dot(A,x)
    norm = np.linalg.norm( (b_ - b) )**2

    return .5 * norm


def grad_obj(x):
    '''
    Gradient of objective function to give direction
    of descent
    '''
    global A 
    global b 

    tmp1 = np.dot( np.dot(A.T,A), x )
    tmp2 = np.dot(A.T,b)

    return tmp1 - tmp2

def sgd_dim(A,b,s):

    global thresh

    x0 = np.ones((A.shape[1],1))
    xk = x0 

    values = [xk]
    c = 10

    for i in range(1,10000):

        # Get next iterate
        xk_1 = xk - (1/(i+c)**s) * grad_obj(xk)

        values.append(xk_1)

        # A check to see if we are close to minimum
        # if abs(obj_func(xk_1)-obj_func(xk)) <= thresh:
        #     # print("[+] Threshold met")
        #     break

        xk = xk_1

    return values

def sgd(A,b,s):

    global thresh

    x0 = np.ones((A.shape[1],1))
    xk = x0 

    values = [xk]

    while True:

        # Get next iterate
        xk_1 = xk - s * grad_obj(xk)

        values.append(xk_1)

        # A check to see if we are close to minimum
        if abs(obj_func(xk_1)-obj_func(xk)) <= thresh:
            # print("[+] Threshold met")
            break

        xk = xk_1

    return values


# Armijo parameters
B = .25
sig = .1
s = 1

# Threshold parameter
thresh = 1e-3

# Get the randomized data
A,b = gen_data()
u, sing, vh = np.linalg.svd(A)
print("")
print("\t[------ Results ------]",end="\n\n")
print("\t[A] Condition #: ",sing[0]/sing[-1],end="\n\n")

# Closed form solution to check algorithm progress
closed_form_solution = np.dot( np.linalg.inv( np.dot(A.T,A) ), np.dot(A.T,b) )
print("\tMin Val: ",obj_func(closed_form_solution),end="\n\n")

############################### CONSTANT STEP-SIZE 

consts = [1e-5,1e-4,1e-3,1e-2,]
const_vals = []
const_objs = []
for c in consts:
    values = sgd(A,b,c)
    obj_vals = [obj_func(xk) for xk in values]
    print("\t [{}] \tConst ==> {}\t # Iterations ==> {}".format(c,obj_vals[-1],len(obj_vals)))
    const_vals.append(values)
    const_objs.append(obj_vals)


plt.figure(figsize=(12,8))

plt.title("Objective Function Progression",fontsize=15)
plt.xlabel("Iterate", fontsize=15)
plt.ylabel("Objective Value - Min Val", fontsize=15)
plt.xlim([0,300])
# plt.ylim([obj_func(closed_form_solution),600])

for p in range(len(consts)):
    plt.plot(range(len(const_objs[p])),const_objs[p] - obj_func(closed_form_solution),label="s = " + str(consts[p]))

plt.legend()
plt.show()

###############################

############################### DIMINISHING STEP-SIZE

B = .25
sig = .1
s = 1

exps = [1.75,2, 2.5,2.75]
print("")
exps_vals = []
exps_objs = []
for e in exps:
    values = sgd_dim(A,b,e)
    obj_vals = [obj_func(xk) for xk in values]
    print("\t[{}] \tDim ==> {}\t # Iterations ==> {}".format(e,obj_vals[-1],len(obj_vals)))

    exps_vals.append(values)
    exps_objs.append(obj_vals)

plt.figure(figsize=(12,8))

plt.title("Objective Function Progression",fontsize=15)
plt.xlabel("Iterate", fontsize=15)
plt.ylabel("Objective Value - Min Val", fontsize=15)
plt.xlim([0,50])
# plt.ylim([obj_func(closed_form_solution),600])

for ex in range(len(exps)):
    plt.plot(range(len(exps_objs[ex])),exps_objs[ex] - obj_func(closed_form_solution),label="s = " + str(exps[ex]))

plt.legend()
plt.show()

###############################


############################### Armijo STEP-SIZE

arm_vals, m_iters = use_armijo(B,sig,s)
arm_obj_vals = [obj_func(xk) for xk in arm_vals]
print("")
print("\tArmijo ==> {}\t\t # Iterations ==> {}".format(arm_obj_vals[-1],len(arm_obj_vals)+m_iters))
print("")
plt.figure(figsize=(12,8))


plt.title("Objective Function Progression",fontsize=15)
plt.xlabel("Iterate", fontsize=15)
plt.ylabel("Objective Value - Min Val", fontsize=15)
plt.xlim([0,20])

plt.plot(range(len(arm_obj_vals)), arm_obj_vals - obj_func(closed_form_solution),label="arm") 
plt.plot(range(len(const_objs[-1])),const_objs[-1] - obj_func(closed_form_solution),label="const ({})".format(consts[-1]))
plt.plot(range(len(exps_objs[0])), exps_objs[0] - obj_func(closed_form_solution), label="dim s = 1.5")

plt.legend()
plt.show()

###############################
