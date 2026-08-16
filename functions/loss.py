from abc import ABC, abstractmethod
from core.matrix import Matrix
from core.vector import Vector
import numpy as np

class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        """Returns scalar loss value."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, y_pred, y_true):
        """Returns gradient of loss with respect to y_pred."""
        raise NotImplementedError


class MSELoss(Loss):

    def forward(self, y_pred, y_true):
        return np.mean((y_pred.data - y_true.data) ** 2)

    def backward(self, y_pred, y_true):
        N = len(y_pred.data)
        grad_data = (2 / N) * (y_pred.data - y_true.data)

        return type(y_pred)(grad_data)


class BCELoss(Loss):
    def __init__(self, eps=1e-15):
        self.eps = eps  # Small offset to prevent log(0)

    def forward(self, y_pred, y_true):
        # Clip y_pred data to [eps, 1 - eps] for stability
        y_pred_clipped = np.clip(y_pred.data, self.eps, 1 - self.eps)
        
        loss = - (y_true.data * np.log(y_pred_clipped) + (1 - y_true.data) * np.log(1 - y_pred_clipped))
        return np.mean(loss)

    def backward(self, y_pred, y_true):
        N = len(y_pred.data)
        y_pred_clipped = np.clip(y_pred.data, self.eps, 1 - self.eps)
        
        grad_data = (1 / N) * ((y_pred_clipped - y_true.data) / (y_pred_clipped * (1 - y_pred_clipped)))
        return type(y_pred)(grad_data)


class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        self.probs = None

    def forward(self, logits, y_true):
        # 1. Compute stable softmax probabilities
        axis = -1 if len(logits.shape) > 1 else 0
        shifted_logits = logits.data - np.max(logits.data, axis=axis, keepdims=True)
        exps = np.exp(shifted_logits)
        self.probs = exps / np.sum(exps, axis=axis, keepdims=True)

        # 2. Compute cross-entropy loss (using small epsilon for log stability)
        eps = 1e-15
        probs_clipped = np.clip(self.probs, eps, 1 - eps)
        loss = -np.sum(y_true.data * np.log(probs_clipped)) / len(logits.data)
        return loss

    def backward(self, logits, y_true):
        # 3. Simple, exact gradient with respect to input logits
        N = len(logits.data)
        grad_data = (self.probs - y_true.data) / N
        return type(logits)(grad_data)
    


#---------------------------------------------------------------------------------------------------------------------------------------------------


loss_fn = SoftmaxCrossEntropy()

# 1. Dummy unnormalized logits for a batch of 2 samples across 3 classes
# Sample 1: Model strongly favors class 0 (logits: 2.0 vs 1.0 vs 0.1)
# Sample 2: Model incorrectly favors class 1 when target is class 2
logits = Matrix([
    [2.0, 1.0, 0.1],
    [0.5, 3.0, 0.2]
])

# 2. One-hot ground-truth targets (Sample 1 -> Class 0, Sample 2 -> Class 2)
y_true = Matrix([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])

# 3. Test Forward Pass (Returns a scalar float loss)
loss_val = loss_fn.forward(logits, y_true)
print(f"Forward Loss: {loss_val:.4f}")

# 4. Test Backward Pass (Returns a Matrix of gradients matching logits shape)
grad = loss_fn.backward(logits, y_true)
print("\nBackward Gradient (dL/dZ):")
print(grad)