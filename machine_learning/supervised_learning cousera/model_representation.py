import numpy as np
import matplotlib.pyplot as plt
plt.style.use(r'C:\Users\Saw\Desktop\machine_learning\week1\deeplearning.mplstyle.txt')
x_train = np.array([1.0,2.0,3.0])
y_train = np.array([300,500,800])
# print(f"x_train: {x_train}")
# print(f"y_train: {y_train}")
# print(f"x_train.shape:{x_train.shape}")
m = x_train.shape[0]
# print(f"number of traning examples: {m}")
m = len(x_train)
# print(f"number of traning examples: {m}")
############################
i = 0
x_i = x_train[i]
y_i = y_train[i]
# print(f"(x^({i}),y^({i}))=({x_i},{y_i})")   

############################
# plt.scatter(x_train,y_train,marker='x',c='g')
# plt.title("housing prince")
# plt.ylabel('Price in 1000s of dollars')
# plt.xlabel('Size (1000 sqft)')
# plt.show()
############################
w = 200
b = 100
# print(f"w: {w}")    
# print(f"b: {b}")  
def compute_model_output(x,w,b):
        """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
        m = x.shape[0]
        f_wb = np.zeros(m)
        for i in range(m):
            f_wb[i]= w*x[i]+b
        return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
############################
w = 200
b = 100
x_i = 1.2
cost_1200sqft = w*x_i+b
print(f"cost_1200sqft: {cost_1200sqft}")
