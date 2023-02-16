import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union
from psychsim.probability import Distribution
from psychsim.pwl import VectorDistributionSet
from model_learning import Trajectory, TeamInfoModelTrajectory

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class ModelLearningResult(object):
    """
    Represents a result of PsychSim model learning for some expert data.
    """

    def __init__(self, data_id: str, trajectories: Union[List[Trajectory], List[TeamInfoModelTrajectory]],
                 stats: Dict[str, np.ndarray]):
        """
        Creates a new result.
        :param str data_id: an identifier for the data for which model learning was performed.
        :param list[Trajectory] trajectories: a list of trajectories, each a sequence of state-action pairs.
        :param dict[str, np.ndarray] stats: a dictionary with relevant statistics regarding the algorithm's execution.
        """
        self.data_id = data_id
        self.trajectories = trajectories
        self.stats = stats


class ModelLearningAlgorithm(ABC):
    """
    An abstract class for PsychSim model learning algorithms, where a *learner* is given a POMDP definition,
    including the world states, observations and beliefs. Given a set of trajectories produced by some target *expert*
    behavior (demonstrations), the goal of the algorithm is to find a PsychSim model such that the learner's behavior
    resulting from using such model best approximates that of the expert.
    The PsychSim model might include the agent's reward function, its available actions (and legality constraints),
    its planning horizon, etc. Some of these elements might be provided and set fixed by the problem's definition,
    while others are set as free parameters to be optimized by the algorithm.
    """

    def __init__(self, label: str, agent_name: str, ):
        """
        Creates a new algorithm.
        :param str label: the label associated with this algorithm (might be useful for testing purposes).
        :param str agent_name: the name of the agent whose behavior that we want to model (the "expert").
        """
        self.label = label
        self.agent_name = agent_name

    @abstractmethod
    def learn(self,
              trajectories: List[Trajectory],
              data_id: Optional[str] = None,
              verbose: bool = False) -> ModelLearningResult:
        """
        Performs model learning by retrieving a PsychSim model approximating an expert's behavior as demonstrated
        through the given trajectories.
        :param list[list[(VectorDistributionSet, Distribution)]] trajectories: a list of trajectories, each containing a
        list (sequence) of state-action pairs demonstrated by an "expert" in the task.
        :param str data_id: an (optional) identifier for the data for which model learning was performed.
        :param bool verbose: whether to show information at each timestep during learning.
        :rtype: ModelLearningResult
        :return: the result of the model learning procedure.
        """
        pass

    @abstractmethod
    def save_results(self, result: ModelLearningResult, output_dir: str, img_format: str):
        """
        Saves the several results of a run of the algorithm to the given directory.
        :param ModelLearningResult result: the results of the algorithm run.
        :param str output_dir: the path to the directory in which to save the results.
        :param str img_format: the format of the images to be saved.
        """
        pass
