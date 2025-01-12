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
        # rewards_comfort = ComfortReward.calculate(self, observations)
        # rewards_electricity = RewardFunction.calculate(self, observations)

        # rewards_comfort = list(map(lambda x: x/48, rewards_comfort))
        # rewards_electricity = list(map(lambda x: x*2.7, rewards_electricity))
        # reward_sum = list(map(np.add, rewards_comfort, rewards_electricity))
        # print("COMFORT", np.mean(rewards_comfort), "ELECTRICITY", np.mean(rewards_electricity))
        # return reward_sum

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

    def calculate2(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        """
        Calculates the custom reward using a simple sum of two normalized rewards:
        the preimplemented environment comfort and the custom reward functions.
        Normalization values determined by simulation on random agents for 10 episodes.

        ! WARNING !
        This function should NOT be used by the user as it does not work lol.

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
        normalized_comfort = [
            (x - min_comfort) / (max_comfort - min_comfort) for x in rewards_comfort
        ]
        min_electricity, max_electricity = -29, 0
        normalized_electricity = [
            (x - min_electricity) / (max_electricity - min_electricity)
            for x in rewards_electricity
        ]
        if np.mean(normalized_electricity) < 0.1:
            print(
                "COMFORT",
                np.mean(normalized_comfort),
                "ELECTRICITY",
                np.mean(normalized_electricity),
            )
        # Combine the normalized rewards
        if np.mean(normalized_comfort) > 1.5 * np.mean(
            normalized_electricity
        ) or 1.5 * np.mean(normalized_comfort) < np.mean(normalized_electricity):
            # print("COMFORT", np.mean(normalized_comfort), "ELECTRICITY", np.mean(normalized_electricity))
            self.counter_bad += 1
        else:
            self.counter_good += 1

        # if self.counter_good % 100 == 0:
        # print(self.counter_bad/self.counter_good) # testing purposes, absolutely unnecessary

        reward_sum = [c + e for c, e in zip(normalized_comfort, normalized_electricity)]

        return reward_sum
