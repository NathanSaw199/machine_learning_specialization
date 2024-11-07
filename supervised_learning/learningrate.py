import math, copy
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use(r'C:\Users\Saw\Desktop\machine learning\machine_learning_specialization\supervised_learning\deeplearning1.mplstyle')
from lab_utils_uni import plt_house_x,plt_contour_wgrad,plt_divergence,plt_gradients

x_train = np.array([1.0,2.0])
y_train = np.array([300,500])

def compute_cost(x,y,w,b):
    #get the number of samples in the array
    m = x.shape[0] 
    #initalize the cost sum
    cost_sum = 0
    #create a loop to go through all the samples
    for i in range(m):
        #calculate the value of the function for each of the samples in the array
        f_wb = w*x[i]+b
        #calculate the cost for each of the samples in the array
        cost = (f_wb -y[i])**2
        #sum the values of the cost
        cost_sum = cost_sum + cost
    # divide the sum of the cost by the number of samples in the array to get the average cost J(w,b)
    total_cost = 1/(2*m)*cost_sum
    return total_cost
def compute_gradient(x,y,w,b):
    #get number of samples in the array
    m=x.shape[0]
    #initialize the sum of the dj dw
    dj_dw =0
    #initialize the sum of the dj db
    dj_db =0
    #create a loop to go through all the samples
    for i in range(m):
        #calculate the value of the function for each of the samples in the array
        f_wb = w*x[i]+b
        #calculate the (fw,b(xi)-yi) *xi
        dj_dw_i = (f_wb-y[i]) * x[i]
        #calculate the fw,b(xi)-yi
        dj_db_i =f_wb-y[i]
        #sum the values of the dj_dw_i
        dj_db += dj_db_i
        #sum the values of the dj_db_i
        dj_dw += dj_dw_i
    #divide the sum of the dj_dw by the number of samples in the array to get the average dj_dw dj(w,b)/dw
    dj_dw =dj_dw/m
    #divide the sum of the dj_db by the number of samples in the array to get the average dj_db dj(w,b)/db
    dj_db =dj_db/m
    return dj_dw,dj_db

# plt_gradients(x_train,y_train,compute_cost,compute_gradient)
# plt.show()
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    #J_history: list to record the cost at each iteration.
    J_history = []
    #p_history: list to record the values of parameters w and b at each iteration.
    p_history = []
    #Initializes b and w with the provided initial values b_in and w_in
    b = b_in
    w = w_in
    #This loop runs for num_iters iterations to perform the gradient descent updates.
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        #Calls gradient_function with the current x, y, w, and b to get the gradients dj_dw (gradient with respect to w) and dj_db (gradient with respect to b)
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        #Update the values of w and b using the gradients dj_dw and dj_db and the learning rate alpha.
        # b= b- alpha*(dj(w,b)/db)
        b = b - alpha * dj_db 
        # w= w- alpha*(dj(w,b)/dw)                           
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # Records the cost and parameters if the number of iterations is less than 100,000 to prevent resource exhaustion.
            #This function call calculates the cost, or the loss, associated with the current values of parameters w (weight) and b (bias). The cost function is typically a measure of how well the model's predictions match the actual target values y given the input features x.This appends the computed cost to the list J_history. By doing this for every iteration, you create a record of how the cost changes as the parameters w and b are updated. 
            J_history.append(cost_function(x, y, w , b))
            #This appends the current list of parameters to the p_history list. Like J_history, recording the history of parameter values allows you to see how w and b evolve over the course of the gradient descent iterations. 
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing

#initialize the values of w and b parameters
w_init = 0
b_init = 0
#some gradient descent settings
iternations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final,b_final,j_hist,p_hist = gradient_descent(x_train,y_train,w_init,b_init,tmp_alpha,iternations,compute_cost,compute_gradient)
print(f"(w,b) found by gradient descent: ( {w_final:8.4f},{b_final:8.4f})")





  
# # plot cost versus iteration  
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
# ax1.plot(j_hist[:100])
# ax2.plot(1000 + np.arange(len(j_hist[1000:])), j_hist[1000:])
# ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
# ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
# ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
# plt.show()