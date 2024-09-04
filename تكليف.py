
#دوال
def greet(name):
    return f"Hello, {name}!"
print(greet("Badee"))

#--------------------------------------

class car:
    def _init_(self,name):
        self.name=name

    def bus(self):
        return f"{self.name}welcom......"

Car = car("Badee")
print(car.bus())

#-----------------------------------------------


import numpy as np

arr = np.array([1,2,3,4,5])
print(arr)

#-------------------------------------------------

import pandas as pd

data = {'name':['badee','ali','mohammed'],'age':[20,23,30]}
df= pd.DataFrame(data)
print(df)


#----------------------------

import cv2

image = cv2.imread('image.ipg')
cv2.imshow('Image',image)
cv2.imwrite('badee.png',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------


class perceptron:
    def _init_(self,learning=0.01,iters=1000):
        self.lr=learning
        self.iters= iters
        self.weights= None
        self.bias = None
    def fit(self,x,y):
        samples,features=x.shape
        self.weights = np.zeros(features)
        self.bias =0

        for _ in range(self.iters):
            for idx,x_i in enumerate(x):
                linear_output=np.dot(x_i,self.weights)+self.bias
                y_predicted = np.where(linear_output >= 0,1,0)
                update = self.lr*(y[idx]-y_predicted)
                self.weights +=update*x_i
                self.bias += update
            def predict(self,x):
                iinear_output = np.dot(x, self.weights) + self.bias
                return np.where(linear_output >= 0, 1, 0)
x= np.array([[1,1],[1,0],[0,1],[0,0]])
y= np.array([1,0,0,0])

model = perceptron(learning_rate=0.1,iters=10)
model.fit(x,y)
predictions=model.predict(x)
print(predictions)