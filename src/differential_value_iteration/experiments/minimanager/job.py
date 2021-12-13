import dataclasses
from typing import Any, Callable, Dict

@dataclasses.dataclass
class Job:
  env_constructor: Callable[[Any, ...], Any]
  env_params: Dict[str, Any]

  agent_constructor: Callable[[Any, ...], Any]
  agent_params: Dict[str, Any]

  run_loop: Callable[[Any, ...], Any]
  run_params: Dict[str, Dict[str, Any]]

  job_id: int
  experiment_id: str

