import numpy as np

class Value:
  def __init__(self, data, _children = (), _op = '', label=''):
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self.grad = 0.0
    self._backward = lambda: None

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    if (not isinstance(other, Value)):
      other = Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0 * out.grad

      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), _op = '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting ints and floats for now"
    out = Value(self.data ** other, (self, ), _op = "**{other}")
    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out
  def __truediv__(self, other):
    return self * (other ** - 1)

  def __rtruediv__(self, other):
    return other * (self ** -1)

  def __rmul__(self, other):
    return self * other

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad

    out._backward = _backward

    return out
  def __neg__(self):
    return self * -1
  def __sub__(self, other):
    return self + (-other)

  def tanh(self):
    x = self.data
    n = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    out = Value(n, (self, ), _op = 'tanh')
    def _backward():
      self.grad += (1 - n**2) * out.grad
    out._backward = _backward
    return out

  def __radd__(self, other): # other + self
    return self + other
  def __rsub__(self, other): # other - self
    return other + (-self)


  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad += 1.0
    for node in reversed(topo):
      node._backward()
