"""Strategies for choosing the order of async state updates."""
import abc
import random


class AsyncManager(abc.ABC):
  """Parent/Helper class for all Async strategies."""
  def __init__(self, num_states: int, start_state: int):
    self.num_states = num_states
    self.start_state = start_state
    self.next_state = start_state

  @abc.abstractmethod
  def update(self, change: float) -> None:
    """Update state of the manager."""
    raise NotImplementedError

  def reset(self) -> None:
    """Reset state of the manager."""
    self.next_state = self.start_state

  @abc.abstractmethod
  def name(self) -> str:
    """Returns a name for the state update strategy."""
    raise NotImplementedError


class RandomAsync(AsyncManager):
  """Updates states in uniformly random order."""
  def __init__(self, num_states: int, start_state: int, seed: int = 42):
    super().__init__(num_states, start_state)
    self.rand_gen = random.Random(seed)
    # Randomize the first update.
    self.update(0.)
    self.seed=seed

  def update(self, change: float) -> None:
    del change
    self.next_state = self.rand_gen.randint(0, self.num_states - 1)

  def reset(self) -> None:
    super().reset()
    self.update(0.)

  def name(self) -> str:
    return f'RandomASyncWithReplace_Seed{self.seed}'

class RandomAsyncWithoutReplacement(AsyncManager):
  """Updates states in uniformly random order."""
  def __init__(self, num_states: int, start_state: int, seed: int = 42):
    super().__init__(num_states, start_state)
    self.rand_gen = random.Random(seed)
    self.seed = seed
    self.all_states = []
    self.replenish()

  def reset(self) -> None:
    super().reset()
    self.all_states.clear()
    self.replenish()

  def replenish(self):
    """Refills the random state list."""
    assert(len(self.all_states)==0)
    self.all_states.extend(list(range(self.num_states)))
    self.rand_gen.shuffle(self.all_states)

  def update(self, change: float) -> None:
    del change
    if len(self.all_states) == 0:
      self.replenish()
    self.next_state = self.all_states.pop()

  def name(self) -> str:
    return f'RandomASyncNoReplace_Seed{self.seed}'


class ConvergeRandomASync(AsyncManager):
  """Random but only switches states after change less than tolerance."""
  def __init__(self, num_states: int, start_state: int, seed: int, tol: float):
    super().__init__(num_states, start_state)
    self.rand_gen = random.Random(seed)
    self.seed = seed
    self.tol = tol

  def update(self, change: float):
    if abs(change) < self.tol:
      self.next_state = self.rand_gen.randint(0, self.num_states - 1)

  def name(self) -> str:
    return f'ConvergeRandomASyncNoReplace_Seed{self.seed}_Tol{self.tol}'


class RoundRobinASync(AsyncManager):
  """Updates states in their numeric order (0, 1, 2, 3... n-1, 0, 1 ...)."""
  def update(self, change: float):
    del change
    self.next_state = (self.next_state + 1) % self.num_states

  def name(self) -> str:
    return 'RoundRobinASync'


class ConvergeRoundRobinASync(AsyncManager):
  """Round Robin but only switches states after change less than tolerance."""
  def __init__(self, num_states: int, start_state: int, tol: float):
    super().__init__(num_states, start_state)
    self.tol = tol

  def update(self, change: float):
    if abs(change) < self.tol:
      self.next_state = (self.next_state + 1) % self.num_states

  def name(self) -> str:
    return f'ConvergeRoundRobinASync_Tol{self.tol}'
