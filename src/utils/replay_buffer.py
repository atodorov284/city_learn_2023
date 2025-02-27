from collections import namedtuple
from typing import Dict

import numpy as np

from typing import List

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """Initialize the replay buffer"""
        self._capacity = capacity
        self._memory = []

    def push(
        self,
        state: List[List[float]],
        action: List[List[float]],
        reward: List[float],
        next_state: List[List[float]],
        done: int,
    ) -> None:
        """Push a transition to the replay buffer"""

        if len(self) >= self._capacity:
            self._memory.pop(0)

        self._memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions from the replay buffer"""
        current_size = len(self)

        batch_size = min(batch_size, current_size)

        indices = np.random.randint(0, current_size, size=batch_size)
        batch = Transition(*zip(*[self._memory[i] for i in indices]))

        return (
            np.array(batch.state),
            np.array(batch.action),
            np.array(batch.reward),
            np.array(batch.next_state),
            np.array(batch.done),
        )

    def __len__(self) -> int:
        """Return the length of the replay buffer"""
        return len(self._memory)
