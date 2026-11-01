import numpy as np;

#x is the maks obtained by student. x=1
#y is the expected output 1 for passing, 0 for fail. y=1
#w is the weight. let w=0.5
def sigmoid(param,deriv=False): #simoid function (0,1)
    if(deriv==True):
        return param*(1-param)  #derivative or slope of sigmoid
    return 1/(1+np.exp(-param))

x = 1
y = 1
w = 0.5

for i in range(0,5):  #five predictions
    print(f"{i} prediction")
    z = w*x  #weight mul by input
    y1 = sigmoid(z)  #predicted output
    print(y1)
    error = y - y1 #error of prediction
    w0 = error + sigmoid(y1,deriv=True) + x #w0 is change in weight
    w = w + w0 #new weight
    


