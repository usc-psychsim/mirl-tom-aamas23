import copy
import numpy as np
from typing import List, Optional
from model_learning.environments.gridworld import Location
from psychsim.agent import Agent
from model_learning import State
from model_learning.environments.objects_gridworld import ObjectsGridWorld
from model_learning.features.linear import LinearRewardVector

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class ObjectsRewardVector(LinearRewardVector):
    """
    Represents a linear reward vector, i.e., a reward function formed by a linear combination of a set of reward
    features.
    """

    def __init__(self,
                 env: ObjectsGridWorld,
                 agent: Agent,
                 feat_matrix: np.ndarray,
                 outer: bool = True,
                 inner: bool = True):
        """
        Creates a new reward vector with the given features.
        :param ObjectsGridWorld env: the objects environment used iby this reward function.
        :param Agent agent: the agent for which to get the location features.
        :param np.ndarray feat_matrix: the feature matrix containing the color features for each environment location,
        shaped (width, height, num_colors [, num_colors]).
        :param bool outer: whether to include features for the presence of objects' outer colors.
        :param bool inner: whether to include features for the presence of objects' inner colors.
        """
        super().__init__([])
        assert inner or outer, 'At least one option "inner" or "outer" must be true'
        assert ((feat_matrix.shape[-1] == env.num_colors) and
                (not (inner and outer) or feat_matrix.shape[-2] == env.num_colors)), \
            'Invalid feature matrix dimensions for provided inner and outer color arguments.'
        self.env = env
        self.x, self.y = env.get_location_features(agent)
        self.feat_matrix = feat_matrix
        self.outer = outer
        self.inner = inner
        self.names: List[str] = [f'Out Color {i}' for i in range(env.num_colors if outer else 0)] + \
                                [f'In Color {i}' for i in range(env.num_colors if inner else 0)]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def get_values(self, state: State) -> np.ndarray:
        """
        Gets an array with the values / counts for each reward feature according to the given state. If the state is
        probabilistic (distribution), then average among all possible states is retrieved, weight by the corresponding
        probabilities.
        :param State state: the world state from which to get the feature value.
        :rtype: np.ndarray
        :return: an array containing the values for each feature.
        """
        # get agent's XY features
        state = copy.deepcopy(state)
        state.collapse({self.x, self.y}, False)  # collapses into single joint distribution

        # collects possible locations and associated probabilities
        locs: List[Location] = []
        probs: List[float] = []
        key = state.keyMap[self.x]
        if key is None:
            # feature values are certain
            x_val = state.certain[self.x]
            y_val = state.certain[self.y]
            locs.append(Location(x_val, y_val))
            probs.append(1.)
        else:
            # feature values are probabilistic, so iterate over values and associated probs
            for dist, prob in state.distributions[key].items():
                x_val = dist[self.x]
                y_val = dist[self.y]
                locs.append(Location(x_val, y_val))
                probs.append(prob)

        # return weighted average of feature vectors
        return np.multiply(
            self.feat_matrix[tuple(np.array(locs).swapaxes(0, 1))],  # shape: (num_locs, num_objects)
            np.array(probs).reshape(len(locs), 1)  # shape: (num_locs, 1)
        ).sum(axis=0)  # shape: (num_objects,)

    def set_rewards(self,
                    agent: Agent,
                    weights_outer: Optional[np.ndarray] = None,
                    weights_inner: Optional[np.ndarray] = None,
                    model: Optional[str] = None):
        """
        Sets a reward to the agent that is a linear combination of the given weights associated with each object color.
        :param Agent agent: the agent to whom the reward is going to be set.
        :param np.ndarray weights_outer: the reward weights associated with each object's outer color.
        :param np.ndarray weights_inner: the reward weights associated with each object's inner color.
        :param str model: the name of the agent's model for whom to set the reward.
        :return:
        """
        agent.setAttribute('R', {}, model)  # make sure to clear agent's reward function
        self.env.set_linear_color_reward(agent,
                                         weights_outer if self.outer else None,
                                         weights_inner if self.inner else None)
