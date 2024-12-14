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
    
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        """ 
        Calculates the custom reward using a simple sum.
        TODO: APPLY WEIGHTING (comfort reward can be even ~50x electricity reward in magnitude),
        which is why I reduced it by 20 for now.
        """
        rewards_comfort = ComfortReward.calculate(self, observations) 
        rewards_electricity = RewardFunction.calculate(self, observations)
        rewards_comfort = list(map(lambda x: x/20, rewards_comfort))
        reward_sum = list(map(np.add, rewards_comfort, rewards_electricity))
        return reward_sum
    
    def calculate2(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        """
        TODO: Use the random agent for multiple episodes to determine what min/max of the
        rewards should be and store it for later
        """
        rewards_comfort = ComfortReward.calculate(self, observations)
        rewards_electricity = RewardFunction.calculate(self, observations)
    
        # Normalize rewards
        min_comfort, max_comfort = 0, 10
        normalized_comfort = [(x - min_comfort) / (max_comfort - min_comfort) for x in rewards_comfort]
        min_electricity, max_electricity = min(rewards_electricity), max(rewards_electricity)
        normalized_electricity = [(x - min_electricity) / (max_electricity - min_electricity) for x in rewards_electricity]
    
        # Combine the normalized rewards
        reward_sum = [c + e for c, e in zip(normalized_comfort, normalized_electricity)]
    
        return reward_sum