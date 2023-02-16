import operator
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List
from psychsim.agent import Agent
from psychsim.pwl import rewardKey, setToConstantMatrix, makeTree, setToFeatureMatrix, KeyedTree, KeyedPlane, \
    KeyedVector, actionKey
from psychsim.world import World
from model_learning import PsychSimType, State

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

COMPARISON_OPS = {'==': operator.eq, '>': operator.gt, '<': operator.lt}


class LinearRewardFeature(ABC):
    """
    Represents a reward feature that can be used in linearly-parameterized reward functions.
    """

    def __init__(self, name: str, normalize_factor: float = 1):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        :param float normalize_factor: the normalization factor for this feature.
        """
        self.name = name
        self.normalize_factor = normalize_factor

    @abstractmethod
    def get_value(self, state: State) -> float:
        """
        Gets the value / count of the feature according to the given state. If the state is probabilistic
        (distribution), then average among all possible states is retrieved, weight by the corresponding probabilities.
        :param State state: the world state from which to get the feature value.
        :rtype: float
        :return: the value of the feature in the given state (expected if state is distribution).
        """
        pass

    @abstractmethod
    def set_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
        """
        Sets a reward to the agent corresponding to the value of the features multiplied by the given weight.
        :param Agent agent: the agent to whom the reward is going to be set.
        :param float weight: the weight associated with the reward feature.
        :param str model: the name of the agent's model for whom to set the reward.
        """
        pass


class ValueComparisonLinearRewardFeature(LinearRewardFeature):
    """
    Represents a boolean reward feature that performs a comparison between a feature and a value, and returns `1` if
    the comparison is satisfied, otherwise returns `0`.
    """

    def __init__(self, name: str, world: World, key: str, value: PsychSimType, comparison: str):
        """
        Creates a new reward feature.
        :param World world: the PsychSim world capable of retrieving the feature's value given a state.
        :param str name: the label for this reward feature.
        :param str key: the named key associated with this feature.
        :param str or int or float value: the value to be compared against the feature to determine its truth (boolean) value.
        :param str comparison: the comparison to be performed, one of `{'==', '>', '<'}`.
        """
        super().__init__(name)
        self.world = world
        self.key = key
        self.value = value
        assert comparison in KeyedPlane.COMPARISON_MAP, \
            f'Invalid comparison provided: {comparison}; valid: {KeyedPlane.COMPARISON_MAP}'
        self.comparison = KeyedPlane.COMPARISON_MAP.index(comparison)
        self.comp_func = COMPARISON_OPS[comparison]

    def get_value(self, state: State) -> float:
        # collects feature value distribution and returns weighted sum
        dist = np.array([[float(self.comp_func(self.world.float2value(self.key, v), self.value)), p]
                         for v, p in state.marginal(self.key).items()])
        return dist[:, 0].dot(dist[:, 1]) * self.normalize_factor

    def set_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
        rwd_key = rewardKey(agent.name)
        tree = {'if': KeyedPlane(KeyedVector({self.key: 1}), self.value, self.comparison),
                True: setToConstantMatrix(rwd_key, 1.),
                False: setToConstantMatrix(rwd_key, 0.)}
        agent.setReward(makeTree(tree), weight * self.normalize_factor, model)


class ActionLinearRewardFeature(ValueComparisonLinearRewardFeature):
    """
    Represents a reward feature that returns `1` if the agent's action is equal to a given action, and `0` otherwise.
    """

    def __init__(self, name: str, agent: Agent, action: str):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        :param Agent agent: the agent whose action is going to be tracked by this feature.
        :param str action: the action tracked by this feature.
        """
        super().__init__(name, agent.world, actionKey(agent.name), action, '==')


class NumericLinearRewardFeature(LinearRewardFeature):
    """
    Represents a numeric reward feature, that returns a reward proportional to the feature's value.
    """

    def __init__(self, name: str, key: str):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        :param str key: the named key associated with this feature.
        """
        super().__init__(name)
        self.key = key

    def get_value(self, state: State) -> float:
        # simply return expectation of marginal
        return state.marginal(self.key).expectation() * self.normalize_factor

    def set_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
        # simply multiply the feature's value by the given weight
        rwd_key = rewardKey(agent.name)
        agent.setReward(KeyedTree(setToFeatureMatrix(rwd_key, self.key)), weight * self.normalize_factor, model)


class LinearRewardVector(object):
    """
    Represents a linear reward vector, i.e., a reward function formed by a linear combination of a set of reward
    features.
    """

    def __init__(self, rwd_features: List[LinearRewardFeature]):
        """
        Creates a new reward vector with the given features.
        :param list[LinearRewardFeature] rwd_features: the list of reward features.
        """
        self.rwd_features = rwd_features
        self.names: List[str] = [feat.name for feat in self.rwd_features]

    def __len__(self):
        return len(self.rwd_features)

    def __iter__(self):
        return iter(self.rwd_features)

    def get_values(self, state: State) -> np.ndarray:
        """
        Gets an array with the values / counts for each reward feature according to the given state. If the state is
        probabilistic (distribution), then average among all possible states is retrieved, weight by the corresponding
        probabilities.
        :param State state: the world state from which to get the feature value.
        :rtype: np.ndarray
        :return: an array containing the values for each feature.
        """
        return np.array([feat.get_value(state) for feat in self.rwd_features])

    def set_rewards(self, agent: Agent, weights: np.ndarray, model: Optional[str] = None):
        """
        Sets rewards to the agent corresponding to the value of each feature multiplied by the corresponding weight.
        :param Agent agent: the agent to whom the reward is going to be set.
        :param np.ndarray weights: the weights associated with each reward feature.
        :param str model: the name of the agent's model for whom to set the reward.
        """
        assert len(weights) == len(self.rwd_features), \
            'Provided weight vector\'s dimension does not match reward features'

        agent.setAttribute('R', {}, model)  # make sure to clear agent's reward function
        for i, weight in enumerate(weights):
            self.rwd_features[i].set_reward(agent, weight, model)
