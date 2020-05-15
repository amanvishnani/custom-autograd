import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union

Arrayable = Union[int, float, list, np.ndarray]


def ensure_arrayable(arrayable: Arrayable):
    if isinstance(arrayable, np.ndarray):
        return arrayable
    return np.array(arrayable)


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on=None) -> None:
        self.data = ensure_arrayable(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "Called Backward on non-requires-grad-tensor"
        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("Grad Must be specified for non-0-tensor")

        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements
    """
    data = t.data.sum()
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            Grad is necessarily a 0-tensor, so each input element
            contributes that much
            """

            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)
