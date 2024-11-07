import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition,plt_stationary,plt_update_onclick,soup_bowl
plt.style.use(r"/Users/nathansaw/Desktop/untitled folder/machine_learning_specialization/supervised_learning/deeplearning1.mplstyle")

# x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
# y_train = np.array([250, 300, 480,  430,   630, 730,])
def compute_cost(x,y,w,b):

     m = x.shape[0] 
     cost_sum = 0 
     for i in range(m): 
          f_wb = w * x[i] + b   
          cost = (f_wb - y[i]) ** 2  
          cost_sum = cost_sum + cost  
     total_cost = (1 / (2 * m)) * cost_sum  
     return total_cost
x_train = np.array([1.0,2.0])
y_train = np.array([300,500])
w = 150
b = 100
# for For x[0] = 1.0, the model's prediction is:
#b=w⋅x[0]+b=150⋅1.0+100=250
#The squared error for this data point is:
#cost=(250−300)**2=(−50)**2 =2500
#For x[1] = 2.0, the model's prediction is:
#b=w⋅x[1]+b=150⋅2.0+100=400
#The squared error for this data point is:
#cost=(400−500)**2=(−100)**2=10000
#Sum of squared errors (cost_sum):
#cost_sum=2500+10000=12500
#Final total cost:
#total_cost= 1/(2⋅2)⋅12500=1/4*12500=3125.0

cur_cost = compute_cost(x_train,y_train,w,b)
print(f"curent cost: {cur_cost}")
# plt.close('all') 
# fig, ax, dyn_items = plt_stationary(x_train, y_train)
# updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
# soup_bowl()