import copy,math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(r"/Users/nathansaw/Desktop/untitled folder/machine_learning_specialization/supervised_learning/deeplearning1.mplstyle")
#This modifies the default print options for numpy arrays, setting the precision to 2 decimal places for clearer output.
np.set_printoptions(precision=2)    ## reduced display precision on numpy arrays
# 2D numpy array representing the features of the training dataset, where each row corresponds to a different example (house) and each column represents a feature (e.g., size, number of rooms, etc.)
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
#1D numpy array representing the target values (house prices) for each example in X_train
y_train = np.array([460, 232, 178])
# data is stored in numpy array/matrix
# print(f"X shape: {X_train.shape}, X Type: {type(X_train)}")
# print(f"X_train : {X_train}")
# print(f"Y shape: {y_train.shape}, Y Type: {type(y_train)}")
# print(f"Y_train : {y_train}")
#initializes the bias term for the linear regression model.
b_init = 785.1811367994083
# initializes the weights (model parameters) for each feature. The length of this array should match the number of features in X_train.
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
#print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
#predict_single_loop function calculates this weighted sum for a single example xxx, which has multiple features, using a loop. It outputs the predicted target value based on the input, weights, and bias.
def predict_single_loop(x,w,b):
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
#Gets the number of features (elements) in x
    n =x.shape[0]
#p will store the sum of the weighted features plus the bias. Initially, it’s set to zero because we haven’t added any weighted values yet.
#Think of p as the variable that will accumulate the predicted value as we go through each feature.
    p=0
#This line starts a loop that will iterate over each feature in x.
#range(n) produces a sequence from 0 to n-1, which means the loop will run once for each feature in x
#Each iteration will access a feature in x and the corresponding weight in w
    for i in range(n):
        #This line calculates the weighted value for the i-th feature of x
        #x[i] is the value of the i-th feature in the input vector x.
        #w[i] is the weight associated with the i-th feature.
        #p_i = x[i] * w[i] multiplies these two values, creating the contribution of this feature to the prediction.
        #This contribution, p_i, is then added to p in the next line.
        p_i = x[i]*w[i]
        #This line adds the weighted feature value p_i to p.
        #p is accumulating the weighted contributions from each feature in x as the loop iterates.
        #So, after each loop iteration, p contains the sum of the products of x[i] * w[i] for all features encountered so far.
        #For example, after the first iteration, p contains the weighted value of the first feature. After the second iteration, it contains the sum of the first and second weighted features, and so on.
        p=p+p_i
        #After the loop completes, p contains the sum of all the weighted feature values.
        #This line adds the bias term b to p. In linear regression, the bias helps adjust the prediction up or down independently of the feature values.
        #Adding the bias term completes the calculation of the prediction.
    p=p+b
    #This line returns the final prediction, which is the sum of the weighted features plus the bias.
    #p is the predicted target value for the input vector x based on the weights w and bias b.
    #x = [2104, 5, 1, 45] (4 features).
    #w = [0.391, 18.75, -53.36, -26.42] (weights for each feature).
    #b = 785.18 (bias term).
  #Initialize: p = 0
  #First iteration (i = 0):
  #p_i = 2104 * 0.391 = 822.064
  #p = 0 + 822.064 = 822.064
  #Second iteration (i = 1):
  #p_i = 5 * 18.75 = 93.75
#p = 822.064 + 93.75 = 915.814
#Third iteration (i = 2):
#p_i = 1 * -53.36 = -53.36
#p = 915.814 - 53.36 = 862.454
#Fourth iteration (i = 3):
#p_i = 45 * -26.42 = -1188.9
#p = 862.454 - 1188.9 = -326.446
#Add Bias:
#p = -326.446 + 785.18 = 458.734
    return p
# get a row from our training data

x_vec =X_train[1,:]
print(f"x_vec shape: {x_vec.shape}, x_vec value: {x_vec}")
# make a prediction
f_wb = predict_single_loop(x_vec,w_init,b_init) 
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

def predict(x,w,b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x,w)+b
    return p
# get a row from our training data
x_vec1 = X_train[1,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
f_wb1 = predict(x_vec1,w_init,b_init)
print(f"f_wb shape {f_wb1.shape}, prediction: {f_wb1}")
#compute_cost calculates the "cost" or "error" of the predictions made by a linear regression model. In linear regression, the cost function (often mean squared error) measures how well the model's predictions match the actual target values. By minimizing this cost function, we improve the model's accuracy.
def compute_cost(X,y,w,b):
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
    #m is set to the number of training examples (rows) in X.
    #X.shape[0] retrieves the number of rows in X, which is the number of examples in the dataset.

    m = X.shape[0]
    #Initializes the cost variable to 0. This variable will accumulate the sum of squared errors for each training example.
    cost = 0
    #This line starts a loop that will iterate over each training example (from 0 to m-1).
    #Each iteration calculates the squared error for one example and adds it to the total cost.
    for i in range(m):
        #X[i] extracts the i-th example from X, which is a 1D array of shape (n,) containing the features for that example.
        #np.dot(X[i], w) calculates the dot product of the i-th example’s features and the weights w, resulting in a scalar value (prediction without the bias).
        #Adding b to this result completes the prediction f_wb_i for the i-th example.
        f_wb_i = np.dot(X[i],w)+b #(n,)(n,) = scalar (see np.dot)
        #This line calculates the squared error for the i-th example by taking the difference between the predicted value f_wb_i and the actual target value y[i].
        #The difference (f_wb_i - y[i]) is squared to penalize larger errors more heavily, following the mean squared error (MSE) formula.
        #This squared error is added to the cumulative cost.
        #After all m examples have been processed, cost contains the total sum of squared errors.


        cost = cost + (f_wb_i - y[i])**2       #scalar
        #divided by 2m to to get the mean squared error.
    cost = cost/(2*m)
    #Returns the computed cost, which indicates how well (or poorly) the model is performing with the current weights w and bias b. A lower cost implies better performance.
    return cost
#compute_cost is called with X_train, y_train, w_init, and b_init.
#cost is printed to show the error of the model’s predictions with the initial weights and bias.

cost = compute_cost(X_train,y_train,w_init,b_init)
print(f"cost: {cost}")
def compute_gradient(X, y, w, b): 
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
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw
#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
# # plot cost versus iteration  
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
# ax1.plot(J_hist)
# ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
# ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
# ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
# ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
# plt.show()