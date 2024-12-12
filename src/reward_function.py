from typing import Any, List, Mapping, Tuple, Union
import numpy as np
from citylearn.data import ZERO_DIVISION_PLACEHOLDER

class RewardFunction:
    r"""Base and default reward function class.

    The default reward is the electricity consumption from the grid at the current time step returned as a negative value.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], **kwargs):
        self.env_metadata = env_metadata

    @property
    def env_metadata(self) -> Mapping[str, Any]:
        """General static information about the environment."""

        return self.__env_metadata
    
    @property
    def central_agent(self) -> bool:
        """Expect 1 central agent to control all buildings."""

        return self.env_metadata['central_agent']
    
    @env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]):
        self.__env_metadata = env_metadata

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        r"""Calculates reward.

        Parameters
        ----------
        observations: List[Mapping[str, Union[int, float]]]
            List of all building observations at current :py:attr:`citylearn.citylearn.CityLearnEnv.
            time_step` that are got from calling :py:meth:`citylearn.building.Building.observations`.

        Returns
        -------
        reward: List[float]
            Reward for transition to current timestep.
        """

        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]

        if self.central_agent:
            reward = [min(sum(net_electricity_consumption)*-1, 0.0)]
        else:
            reward = [min(v*-1, 0.0) for v in net_electricity_consumption]

        return reward

class MARL(RewardFunction):
    """MARL reward function class.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """

    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        district_electricity_consumption = sum(net_electricity_consumption)
        building_electricity_consumption = np.array(net_electricity_consumption, dtype=float)*-1
        reward_list = np.sign(building_electricity_consumption)*0.01*building_electricity_consumption**2*np.nanmax([0, district_electricity_consumption])

        if self.central_agent:
            reward = [reward_list.sum()]
        else:
            reward = reward_list.tolist()
        
        return reward

class IndependentSACReward(RewardFunction):
    """Recommended for use with the `SAC` controllers.
    
    Returned reward assumes that the building-agents act independently of each other, without sharing information through the reward.

    Parameter
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        reward_list = [min(v*-1**3, 0) for v in net_electricity_consumption]

        if self.central_agent:
            reward = [sum(reward_list)]
        else:
            reward = reward_list

        return reward
    
    
class ComfortReward(RewardFunction):
    """Reward for occupant thermal comfort satisfaction.

    The reward is the calculated as the negative delta between the setpoint and indoor dry-bulb temperature raised to some exponent
    if outside the comfort band. If within the comfort band, the reward is the negative delta when in cooling mode and temperature
    is below the setpoint or when in heating mode and temperature is above the setpoint. The reward is 0 if within the comfort band
    and above the setpoint in cooling mode or below the setpoint and in heating mode.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    band: float, default = 2.0
        Setpoint comfort band (+/-).
    lower_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is above setpoint upper
        boundary or heating mode but temperature is below setpoint lower boundary.
    higher_exponent: float, default = 2.0
        Penalty exponent for when in cooling mode but temperature is below setpoint lower
        boundary or heating mode but temperature is above setpoint upper boundary.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], band: float = None, lower_exponent: float = None, higher_exponent: float = None):
        super().__init__(env_metadata)
        self.band = band
        self.lower_exponent = lower_exponent
        self.higher_exponent = higher_exponent

    @property
    def band(self) -> float:
        return self.__band
    
    @property
    def lower_exponent(self) -> float:
        return self.__lower_exponent
    
    @property
    def higher_exponent(self) -> float:
        return self.__higher_exponent
    
    @band.setter
    def band(self, band: float):
        self.__band = 2.0 if band is None else band

    @lower_exponent.setter
    def lower_exponent(self, lower_exponent: float):
        self.__lower_exponent = 2.0 if lower_exponent is None else lower_exponent

    @higher_exponent.setter
    def higher_exponent(self, higher_exponent: float):
        self.__higher_exponent = 3.0 if higher_exponent is None else higher_exponent

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []

        for o in observations:
            heating_demand = o.get('heating_demand', 0.0)
            cooling_demand = o.get('cooling_demand', 0.0)
            heating = heating_demand > cooling_demand
            indoor_dry_bulb_temperature = o['indoor_dry_bulb_temperature']
            set_point = o['indoor_dry_bulb_temperature_set_point']
            lower_bound_comfortable_indoor_dry_bulb_temperature = set_point - self.band
            upper_bound_comfortable_indoor_dry_bulb_temperature = set_point + self.band
            delta = abs(indoor_dry_bulb_temperature - set_point)
            
            if indoor_dry_bulb_temperature < lower_bound_comfortable_indoor_dry_bulb_temperature:
                exponent = self.lower_exponent if heating else self.higher_exponent
                reward = -(delta**exponent)
            
            elif lower_bound_comfortable_indoor_dry_bulb_temperature <= indoor_dry_bulb_temperature < set_point:
                reward = 0.0 if heating else -delta

            elif set_point <= indoor_dry_bulb_temperature <= upper_bound_comfortable_indoor_dry_bulb_temperature:
                reward = -delta if heating else 0.0

            else:
                exponent = self.higher_exponent if heating else self.lower_exponent
                reward = -(delta**exponent)

            reward_list.append(reward)

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward


class CustomRewardFunction(ComfortReward):
    r"""Custom reward function class.
    Combines :py:class:`citylearn.reward_function.ComfortReward (comfort of the occupants)` 
    and :py:class:`citylearn.reward_function.RewardFunction (electricity consumption)`

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    **kwargs : dict
        Other keyword arguments for custom reward calculation.
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], **kwargs):
        super().__init__(env_metadata, **kwargs)
    
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        # fix lol
        return super().calculate(observations) + RewardFunction.calculate(self, observations)