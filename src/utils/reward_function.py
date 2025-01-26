from typing import Any, List, Mapping, Union
import numpy as np
from citylearn.reward_function import (
    RewardFunction,
    ComfortReward,
)


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
        """Initialize the custom reward function."""
        super().__init__(env_metadata, **kwargs)
        self.counter_good = 0
        self.counter_bad = 0

    def calculate(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        """
        Calculates the custom reward using a weighted average.
        Coefficients found by observing the reward distributions,
        scaling them to around -100;0. For more information on the
        coefficients, see Methods section of the paper.
        Args:
            observations (List[Mapping[str, Union[int, float]]]): List of
                observations from the environment.
        Returns:
            List[float]: A list of rewards.
        """
        rewards_comfort = self._comfort_score(observations)
        rewards_electricity = self._electricity_score(observations)

        rewards = rewards_comfort + rewards_electricity
        return rewards

    def _electricity_score(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        electricity_score = RewardFunction.calculate(self, observations)
        return list(map(lambda x: -(x * x), electricity_score))

    def _comfort_score(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        reward_list = []

        tolerance = 2.0  # How many degrees can the temperature be off before it is considered uncomfortable.

        for obs in observations:
            indoor_temperature = obs["indoor_dry_bulb_temperature"]
            set_point = obs["indoor_dry_bulb_temperature_set_point"]

            temp_difference = np.abs(indoor_temperature - set_point)

            if temp_difference > tolerance:
                reward = -((temp_difference) ** 2)

            else:
                reward = 0

            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward
