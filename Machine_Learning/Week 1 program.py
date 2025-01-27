import math 
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

plt.plot(x_train, y_train, 'o',marker='x',ms=20,mec='r',mfc='k')
plt.show()

m = x_train.shape[0]

def compute_cost(x, y, w, b):
    j=0
    for i in range(m):
        j+=1/(2*m)*(w*x[i]+b-y[i])**2
    return j

def compute_gradient(x, y, w, b):
    dw_x=0
    for i in range(m):
        dw_x+=(1/m)*(w*x[i]+b-y[i])*x[i]
    db_x=0
    for i in range(m):
        db_x+=(1/m)*(w*x[i]+b-y[i])
    return dw_x,db_x


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w=w_in
    b=b_in
    J_history=[]
    p_history=[]
    for i in range(num_iters):
        #print(f"hi{i}")
        dw,db=compute_gradient(x, y, w, b)
        w-=alpha*dw
        b-=alpha*db
        if i < 100000: # prevent resource exhaustion
            J_history.append(compute_cost(x, y, w , b))
            p_history.append([w,b])
         # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
            print(f'Iteration {i}: Cost {J_history[-1]} ',f'dj_dw: {dw}, dj_db: {db}  ',f'w: {w}, b:{b}')
    return w,b,J_history,p_history


w_final,b_final,J_history_final,p_history_final=gradient_descent(x_train, y_train, 0, 0, 1e-2, 10000)
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
#axi are the axes of the individual subplots
#ax1=initial subplot of j 
#ax2=final subplot of j
ax1.plot(J_history_final[:100])
ax2.plot( 1000+np.arange(len(J_history_final[1000:])), J_history_final[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

#In graph a1 the J changes so drasticaly just for 100 steps
#In graph a2 the J changes less drastically even for 10,000 steps
