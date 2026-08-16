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

    def _sameClass (self, original, data):
        return type(original)(data)


class ReLU(Activation):
    def forward(self, x):
        result = np.maximum(0, x.data) # i took help online for this one. 
        return self._sameClass(x, result) # this was my idea, i knew that type gives us the class name of the object. and i had to look up if we can use it to make the object. 

    def derivative(self, x):
        result = (x.data > 0).astype(float) 
        return self._sameClass(x, result)


class Sigmoid(Activation):

    def __init__(self):
        self.cache = None

    def forward(self, x):
        # 1 / (1 + e^-x). i looked this up. as i did not know what the sigmoid function was.
        result = 1 / ( 1 + np.exp(-(x.data))) # i got the logic right. gotta say, numpy is a very useful library. especially with operator overloading. as my matrix and vectors are iterable.
        self.cache = self._sameClass(x, result)
        return self.cache

    def derivative(self, x):
        # s * (1 - s )
        result = self.cache.data * ( 1 - self.cache.data)
        return type(self.cache)(result)


class Tanh(Activation):

    def __init__(self): # got the idea of using instance variable to store the once computed data. 
        self.cache = None

    def forward(self, x):
        # e^x - e^-x / e^x + e^-x 
        result = np.tanh(x.data)
        self.cache = type(x)(result)
        return self.cache

    def derivative(self, x):
        # 1 - s^2
        result = 1 - (self.cache.data ** 2)
        return type(self.cache)(result)

class LeakyReLU(Activation):

    def __init__(self, alpha):
        # keeping leaky relu modular enough that we can try for different values of alpha. and if required also train it.
        self.alpha = alpha 

    
    def forward(self, x):
        result = np.maximum((self.alpha * x.data ), x.data) 
        return self._sameClass(x, result)

    def derivative(self, x):
        result = np.where(x.data > 0, 1.0, self.alpha) # i made an elaborate for loop for this, it got crashed at run time. but it turns out there is a numpy function for this as well.. 
        return self._sameClass(x, result)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

# test

relu = ReLU()

x = Vector(-2,0,3)
y = relu.forward(x)

print(y)

x = Matrix([[-1,3], [4,-6]])
y = relu.forward(x)

print(y)


# ==========================================
# 1. SIGMOID TEST
# ==========================================
sigmoid = Sigmoid()

# Test 1A: Vector with negative, zero, and positive values
v_sig = Vector(-5.0, 0.0, 5.0)
out_sig = sigmoid.forward(v_sig)
grad_sig = sigmoid.derivative(v_sig)

print("--- SIGMOID VECTOR TEST ---")
print("Input:     ", v_sig)
print("Forward:   ", out_sig)  # Expect approx: [0.0067, 0.5, 0.9933]
print("Derivative:", grad_sig)  # Expect approx: [0.0066, 0.25, 0.0066]

# Key Sigmoid Behaviors to Notice:
# 1. Forward pass at x = 0 is ALWAYS exactly 0.5.
# 2. Derivative peaks at x = 0 with a maximum value of 0.25.
# 3. For extreme inputs (x = -5 or x = 5), the derivative drops close to 0 (Vanishing Gradient).


# ==========================================
# 2. TANH TEST
# ==========================================
tanh = Tanh()

# Test 2A: Vector centered around 0
v_tanh = Vector(-2.0, 0.0, 2.0)
out_tanh = tanh.forward(v_tanh)
grad_tanh = tanh.derivative(v_tanh)

print("\n--- TANH VECTOR TEST ---")
print("Input:     ", v_tanh)
print("Forward:   ", out_tanh)  # Expect approx: [-0.9640, 0.0, 0.9640]
print("Derivative:", grad_tanh)  # Expect approx: [0.0707, 1.0, 0.0707]

# Key Tanh Behaviors to Notice:
# 1. Forward pass at x = 0 is ALWAYS exactly 0.0 (Zero-centered).
# 2. Derivative peaks at x = 0 with a maximum value of 1.0 (4x stronger than Sigmoid).
# 3. Output range spans symmetrically between -1 and 1.


# ==========================================
# 3. LEAKY RELU TEST
# ==========================================
# Testing with custom alpha = 0.1 to make negative scaling easy to visually verify
leaky = LeakyReLU(0.1)


# Test 3A: Matrix input with positive and negative batches
m_leaky = Matrix([[-10.0, 5.0], [0.0, -2.0]])

out_leaky = leaky.forward(m_leaky)
grad_leaky = leaky.derivative(m_leaky)

print("\n--- LEAKY RELU MATRIX TEST ---")
print("Input:\n", m_leaky)
print(
    "Forward:\n", out_leaky
)  # Expect: Matrix([[-1.0, 5.0], [0.0, -0.2]]) (scaled by alpha=0.1)
print(
    "Derivative:\n", grad_leaky
)  # Expect: Matrix([[0.1, 1.0], [0.1, 0.1]]) (alpha for x<=0, 1 for x>0)

# Key LeakyReLU Behaviors to Notice:
# 1. Positive inputs (5.0) remain unchanged in both forward (5.0) and derivative (1.0).
# 2. Negative inputs (-10.0) are scaled down by alpha (-1.0) instead of being flattened to 0.
# 3. Derivative for negative inputs is non-zero (0.1), preventing dead neurons.