import numpy as np
from core.matrix import Matrix
from layers.dense import Dense
from functions.loss import SoftmaxCrossEntropy

# 1. Dummy batch of 2 samples with 4 features each
X = Matrix([
    [1.0, 2.0, -1.0, 0.5],
    [0.5, -1.5, 2.0, 1.0]
])

# Targets: 2 samples, 3 classes
Y_true = Matrix([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

# 2. Layer and Loss setup
dense = Dense(in_features=4, out_features=3)
loss_fn = SoftmaxCrossEntropy()

# 3. Forward Pass
logits = dense.forward(X)
loss = loss_fn.forward(logits, Y_true)
print(f"Initial Loss: {loss:.4f}")

# 4. Backward Pass
dZ = loss_fn.backward(logits, Y_true)
dX = dense.backward(dZ)

print("Gradient wrt Weights (dW) Shape:", dense.dW.shape)  # Expected: (4, 3)
print("Gradient wrt Input (dX) Shape:", dX.shape)            # Expected: (2, 4)