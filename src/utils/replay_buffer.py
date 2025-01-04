from collections import namedtuple
from typing import Dict

import numpy as np

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """Initialize the replay buffer"""
        self._capacity = capacity
        self._memory = []

    def push(
        self, state: list, action: list, reward: float, next_state: list, done: int
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


if __name__ == "__main__":
    buffer = ReplayBuffer(10)
    buffer.push([1, 2], [3, 4], 5, [6, 7])
    buffer.push([2, 2], [3, 4], 0, [6, 7])
    buffer.push([3, 2], [3, 5], 1, [6, 7])
    print(buffer.sample(3))
