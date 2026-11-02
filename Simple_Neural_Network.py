import numpy as np

def sigmoid(x,deriv=False):                         #sigmoid function
    if(deriv==True):
        return x*(1-x)                              #return the slope of sigmoid function
    return 1/(1+np.exp(-x))                         #returns sigmoid function result
    
x = np.array([[0,0,1]                               #input array of 4 trainning sets
             ,[0,1,1]
             ,[1,0,1]
             ,[1,1,1]])
             
y = np.array([[0]                                   #expected output for 4 trainning sets above
             ,[0]
             ,[1]
             ,[1]])
             
weight = 2*np.random.random((3,1))-1                #weights or hidden layer
#print(weight)
for i in range(1000):                               #1000 times optimizing output
    l0 = x                                          #storing input in lo
    l1 = sigmoid(np.dot(l0,weight))                 #debug
    #print(f"{i} prediction\n")
    #print(l1)
    #print("\n")
    error = y - l1                                  #error(expected-predicted)
    l1_change = error*sigmoid(l1,deriv=True)        #this line and below line mul's input(x),error,slope of sigmoid
    l1_change = np.dot(l0.T,l1_change)              #here lo is of 4X3 so change to 3X4,l1_change is of 4X1
    weight = weight + l1_change                     #backpropogation or updating weight
    
print("Output After Trainning\n")
print(l1)                                           #The final optimized prediction