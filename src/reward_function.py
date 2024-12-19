from typing import Any, List, Mapping, Tuple, Union
import numpy as np
from citylearn.data import ZERO_DIVISION_PLACEHOLDER
from citylearn.reward_function import RewardFunction, ComfortReward, IndependentSACReward, MARL

    

class CustomRewardFunction(ComfortReward):
    """Custom reward function class.
    Combines :py:class:`citylearn.reward_function.ComfortReward (comfort of the occupants)` 
    and :py:class:`citylearn.reward_function.RewardFunction (electricity consumption)`

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], **kwargs) -> None:
        """ Initialize the custom reward function. """
        super().__init__(env_metadata, **kwargs)
        self.counter_good = 0
        self.counter_bad = 0
    
    def calculate2(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        """ 
        Calculates the custom reward using a weighted average.
        ! WARNING !
        This function should NOT be used by the user unless they stricly want to 
        apply a weighted average of the two rewards.
        Args:
            observations (List[Mapping[str, Union[int, float]]]): List of 
                observations from the environment.
        Returns:
            List[float]: A list of rewards.
        """
        rewards_comfort = ComfortReward.calculate(self, observations) 
        rewards_electricity = RewardFunction.calculate(self, observations)
        rewards_comfort = list(map(lambda x: x/20, rewards_comfort))
        reward_sum = list(map(np.add, rewards_comfort, rewards_electricity))
        return reward_sum
    
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        """
        Calculates the custom reward using a simple sum of two normalized rewards:
        the preimplemented environment comfort and the custom reward functions.
        Normalization values determined by simulation on random agents for 10 episodes.
        Args:
            observations (List[Mapping[str, Union[int, float]]]): List of observations
                from the environment.
        Returns:
            List[float]: A list of rewards.
        """
        rewards_comfort = ComfortReward.calculate(self, observations)
        rewards_electricity = RewardFunction.calculate(self, observations)
    
        # Normalize rewards
        min_comfort, max_comfort = -4500, 0
        normalized_comfort = [(x - min_comfort) / (max_comfort - min_comfort) for x in rewards_comfort]
        min_electricity, max_electricity = -29, 0
        normalized_electricity = [(x - min_electricity) / (max_electricity - min_electricity) for x in rewards_electricity]
    
        # Combine the normalized rewards
        if np.mean(normalized_comfort) > 1.5 * np.mean(normalized_electricity) or 1.5 * np.mean(normalized_comfort) < np.mean(normalized_electricity):
            # print("COMFORT", np.mean(normalized_comfort), "ELECTRICITY", np.mean(normalized_electricity))
            self.counter_bad += 1
        else:
            self.counter_good += 1

        # if self.counter_good % 100 == 0:
            # print(self.counter_bad/self.counter_good) # testing purposes, absolutely unnecessary

        reward_sum = [c + e for c, e in zip(normalized_comfort, normalized_electricity)]
    
        return reward_sum