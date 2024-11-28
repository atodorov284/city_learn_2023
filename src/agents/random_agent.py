from typing import Any, List

from citylearn.citylearn import CityLearnEnv

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env: CityLearnEnv, **kwargs: Any) -> None:
        '''
        Initialize the environment and set up the random agent.
        '''
        super().__init__(env, **kwargs)

    def predict(
        self, observations: List[List[float]], deterministic: bool = None
    ) -> List[List[float]]:
        """Provide actions for current time step.

        Return randomly sampled actions from `action_space`.

        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Whether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[List[float]]
            Action values
        """
        actions = [list(s.sample()) for s in self.action_space]
        self.actions = actions
        self.next_time_step()
        return actions
