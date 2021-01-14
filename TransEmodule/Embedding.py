import numpy as np

class Embedding:
  """
    Embedding class - initialize random matrix
  """
  def __init__(self, vector=None, min=None, max=None, shape=None):
    if vector is None:
      self._vector = np.random.uniform(min, max, shape)
    else:
      self._vector = np.array(vector)

  @property
  def vector(self):
    return self._vector

  @vector.setter
  def vector(self, vector):
    self._vector = vector

  def normalize(self):
    if self._vector.ndim == 2:
      self._vector = self._vector/np.linalg.norm(self._vector, axis=1)[:,None]
    elif self._vector.ndim == 1:
      self._vector = self._vector/np.linalg.norm(self._vector)
    else:
       raise Exception('Vector should have 1 or 2 dimension')