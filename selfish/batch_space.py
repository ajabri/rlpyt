import numpy as np
from .space import Space


class Batch(Space):
    """
    A batch (i.e., product) of identical simpler spaces
    Example usage:
    self.observation_space = spaces.Batch((spaces.Discrete(3) for _ in range(N)))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        for space in spaces:
            assert isinstance(space, Space), "Batch of the tuple must be homogenoeous instances of gym.Space"
        super(Batch, self).__init__(None, None)

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces]

    def sample(self):
        return np.concatenate([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space,part) in zip(self.spaces,x))

    def __repr__(self):
        return "Batch(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [space.to_jsonable([sample[i] for sample in sample_n]) \
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        return [sample for sample in zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])]

    def __getitem__(self, index):
        return self.spaces[index]

    def __len__(self):
        return len(self.spaces)
      
    def __eq__(self, other):
        return isinstance(other, Batch) and self.spaces == other.spaces