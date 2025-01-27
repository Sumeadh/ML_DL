import copy, math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

m = np.shape(x_train)[0]
n = np.shape(x_train)[1]
alpha = 10e-8
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    cost = 0.0
    f_wb_i=np.zeros(m)
    for i in range(m):
        f_wb_i[i] = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
    cost =  (f_wb_i - y)**2  # scalar
    cost = np.sum(cost) /m# scalar
    return cost


def derivative(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.random.rand(n,)
    dj_db = 0.
 
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    return dj_dw,dj_db

def train(x,y,w,b,alpha,iter,dw,db):
    j_history = []
    p_history=[]

    for i in range(iter):
        w=w-(alpha/m)*dw
        b=b-(alpha/m)*db
        if i < 1000:  # prevent resource exhaustion
            j_history.append(compute_cost(x, y, w , b))
            p_history.append([w,b])
        if i%math.ceil(iter/10)==0:
            print(f"iteration:{i}",f"w:{w}",f"b:{b}",f"cost:{compute_cost(x,y,w,b)}",f"dw:{dw/m}",f"db:{db/m}")
        #print(db,b)
        dw,db=derivative(x,y,w,b)
    return w,b,j_history,p_history
Dw, Db = derivative(x_train, y_train,w_init,b_init)
w_final, b_final, j_final, p_final = train(x_train, y_train, w_init, b_init, alpha, 1000, Dw, Db)
print(f"w_fianl:{w_final}",f"b_final:{b_final}")
plt.plot(p_final[0:100][0][0],j_final[0:100])
plt.xlabel('Iteration Steps')
plt.ylabel('Cost')
plt.show()

