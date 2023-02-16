from typing import Union, List, Dict
from psychsim.probability import Distribution
from psychsim.pwl import VectorDistributionSet
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


# types
class StateActionPair(object):
    """
    Represents a state-action pair which can have associated a probability.
    """

    def __init__(self, world: World, action: Distribution, prob: float = 1.):
        """
        Creates a new state-action pair.
        :param World world: the world containing the state.
        :param Distribution action: the (stochastic) action selection associated with the state of the world.
        :param float prob: the probability with which the state of the world was selected (for stochastic states).
        """
        self.world: World = world
        self.action: Distribution = action
        self.prob: float = prob


class TeamStateActionPair(object):
    def __init__(self, world: World, action: Dict[str, Distribution], prob: float = 1.):
        self.world: World = world
        self.action: Dict[str, Distribution] = action
        self.prob: float = prob


class TeamStateinfoActionModelTuple(object):
    def __init__(self, state: VectorDistributionSet, action: Dict[str, Distribution],
                 model_dist: Distribution, prob: float = 1.):
        self.state: VectorDistributionSet = state
        self.action: Dict[str, Distribution] = action
        self.model_dist: Distribution = model_dist
        self.prob: float = prob

PsychSimType = Union[float, int, str]
State = VectorDistributionSet
Trajectory = List[StateActionPair]  # list of state (world) - action (distribution) pairs
TeamTrajectory = List[TeamStateActionPair]
TeamInfoModelTrajectory = List[TeamStateinfoActionModelTuple]
