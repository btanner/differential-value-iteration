"""Strategies for choosing the order of async state updates."""
import random


class AsyncManager:
  """Parent/Helper class for all Async strategies."""
  def __init__(self, num_states: int, start_state: int):
    self.num_states = num_states
    self.start_state = start_state
    self.next_state = start_state

  def update(self, change: float) -> None:
    pass

  def reset(self) -> None:
    self.next_state = self.start_state


class RandomAsync(AsyncManager):
  """Updates states in uniformly random order."""
  def __init__(self, num_states: int, start_state: int, seed: int = 42):
    super().__init__(num_states, start_state)
    self.rand_gen = random.Random(seed)

  def update(self, change: float) -> None:
    del change
    self.next_state = self.rand_gen.randint(0, self.num_states - 1)


class ConvergeRandomASync(AsyncManager):
  """Random but only switches states after change less than tolerance."""
  def __init__(self, num_states: int, start_state: int, seed: int, tol: float):
    super().__init__(num_states, start_state)
    self.rand_gen = random.Random(seed)
    self.tol = tol

  def update(self, change: float):
    if abs(change) < self.tol:
      self.next_state = self.rand_gen.randint(0, self.num_states - 1)


class RoundRobinASync(AsyncManager):
  """Updates states in their numeric order (0, 1, 2, 3... n-1, 0, 1 ...)."""
  def update(self, change: float):
    del change
    self.next_state = (self.next_state + 1) % self.num_states


class ConvergeRoundRobinASync(AsyncManager):
  """Round Robin but only switches states after change less than tolerance."""
  def __init__(self, num_states: int, start_state: int, tol: float):
    super().__init__(num_states, start_state)
    self.tol = tol

  def update(self, change: float):
    if abs(change) < self.tol:
      self.next_state = (self.next_state + 1) % self.num_states