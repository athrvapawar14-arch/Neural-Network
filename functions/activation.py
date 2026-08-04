"""

1. it must return a vector. cause i think a vector will be more easy to pass around. 

2. even this output must be a vector/matrix object. cause that would allow uniformity of data throughout the whole propogation. 

3. shouldn't it be both? i mean if the input is not modified, then how can we say the network is learning something? 
and the returned value must also be new, or else what would be the point in sending? or is it the training phase here output data 
must not be modified?


i honestly have no deep idea about these ML terms. these are just my gusses. 

"""
from abc import ABC, abstractmethod
from core.matrix import Matrix
from core.vector import Vector
import numpy as np


class Activation(ABC): 
    # this one i took help as i knew what the abstract classes were. but i never properly used them. now i know. they are like raw blue print of a class. and any class that inherits them must have such methods in them. 


    @abstractmethod
    def forward(self, x):

        raise NotImplementedError
        

    @abstractmethod
    def derivative(self, x):

        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        result = np.maximum(0, x.data) # i took help online for this one. 
        return type(x)(result) # this was my idea, i knew that type gives us the class name of the object. and i had to look up if we can use it to make the object. 

    def derivative(self, x):
        result = (x.data > 0).astype(float) 
        return type(x)(result)



# test

relu = ReLU()

x = Vector(-2,0,3)
y = relu.forward(x)

print(y)

x = Matrix([[-1,3], [4,-6]])
y = relu.forward(x)

print(y)