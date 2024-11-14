from lab_utils_common2 import plot_data, plt_tumor_data, sigmoid, compute_cost_logistic,dlc
from plt_quad_logistic import plt_quad_logistic, plt_prob
import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(r"C:\Users\Saw\Desktop\machine learning\machine_learning_specialization\supervised_learning\deeplearning1.mplstyle")
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
# fig,ax = plt.subplots(1,1,figsize=(4,4))
# plot_data(X_train, y_train, ax)
# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel('$x_1$', fontsize=12)
# ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()
def compute_gradient_logistic(X,y,w,b):
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    #m and n are assigned the dimensions of X, where m is the number of training examples and n is the number of features per example.
    m,n = X.shape
    #dj_dw is initialized as an array of zeros with the same length as the number of features (n). This will hold the gradient of the cost function with respect to each element in w.
    dj_dw = np.zeros(n,)
    #dj_db is initialized to 0. It will store the gradient of the cost function with respect to the bias b.
    dj_db = 0
    #This loop iterates over each individual example in the dataset. m is the total number of training examples in the dataset. Each iteration of this loop processes one training example at a time, denoted by the index i.
    for i in range(m):
        #f_wb_i computes the logistic regression prediction using the sigmoid function and the current weights and bias.
        #np.dot(X[i], w): This expression performs the dot product between the feature vector of the ith training example, X[i], and the weight vector w. This dot product is a common operation in many machine learning algorithms, combining the input features linearly using the weights.
        #+ b: The result of the dot product is then added to the bias b. This represents the linear combination of inputs and weights, offset by the bias, forming the input to the sigmoid function.
        #sigmoid(...): The sigmoid function is applied to the result of the linear combination. The sigmoid function outputs a value between 0 and 1, which is ideal for binary classification as it can be interpreted as the probability of the input belonging to the positive class (label 1).
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        #Err_i: This is the error for the ith example. It is calculated by subtracting the actual target value y[i] from the predicted probability f_wb_i. If the prediction is perfect, Err_i would be zero. A positive value indicates an overestimation by the model, and a negative value indicates an underestimation
        #Err_i calculates the error by subtracting the actual target y[i] from the predicted f_wb_i.
        Err_i = f_wb_i - y[i]
        #This loop iterates over each feature j of the training example i. n is the number of features in the dataset. The loop accumulates the gradient of the cost function with respect to each weight w[j] in dj_dw.
        #dj_db is updated by adding the error, accumulating the gradient of the cost with respect to the bias.
        for j in range(n):
            #dj_dw[j]: This is the jth component of the gradient vector dj_dw, which pertains to the jth weight w[j].
            #Err_i * X[i,j]: The product of the error Err_i and the jth feature of the ith example X[i,j] is computed. This product is a part of the gradient calculation for weight w[j], indicating how the error would change with a small change in w[j].
            # The result is added to the existing value of dj_dw[j], accumulating the gradient over all examples for the jth weight.
            error_w = Err_i * X[i,j]            
            dj_dw[j] = dj_dw[j] + error_w
            #the error Err_i is added to dj_db, which accumulates the gradient of the cost with respect to the bias over all examples.
        dj_db = dj_db + Err_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_db, dj_dw
        
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
    # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)  
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db         
         # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")


fig,ax = plt.subplots(1,1,figsize=(5,4))
# plot the probability 
plt_prob(ax, w_out, b_out)

# Plot the original data
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# Plot the decision boundary
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()
      
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()
w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )