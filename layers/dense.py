import numpy as np
from core.matrix import Matrix
from core.vector import Vector

import numpy as np
from core.matrix import Matrix



class Dense:
    def __init__(self, in_features: int, out_features: int):
        # He Normal Initialization: np.random.randn * sqrt(2 / in_features)
        self.W = Matrix(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self.b = Matrix(np.zeros((1, out_features)))
        
        # Gradients for parameter updates during optimization
        self.dW = None
        self.db = None
        
        # Cache input X for backward pass
        self.x_cache = None

    def forward(self, x: Matrix) -> Matrix:
        self.x_cache = x
        # Z = X @ W + b (Uses matrix matmul and broadcasting addition)
        return (x @ self.W) + self.b

    def backward(self, dZ: Matrix) -> Matrix:
        # 1. dW = X^T @ dZ
        self.dW = self.x_cache.T @ dZ
        
        # 2. db = sum over rows of dZ (keepdims=True maintains (1, out_features) shape)
        self.db = dZ.sum(axis=0, keepdims=True)
        
        # 3. dX = dZ @ W^T (Passed to the previous layer)
        dX = dZ @ self.W.T
        return dX