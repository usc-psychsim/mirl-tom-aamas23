import logging
import os
import numpy as np
from timeit import default_timer as timer
from typing import Optional, List
from psychsim.agent import Agent
from model_learning import TeamInfoModelTrajectory
from model_learning.algorithms import ModelLearningAlgorithm, ModelLearningResult
from model_learning.features import expected_feature_counts, estimate_feature_counts_with_inference
from model_learning.features.linear import LinearRewardVector
from model_learning.util.plot import plot_evolution

__author__ = 'Pedro Sequeira and Haochen Wu'
__email__ = 'pedrodbs@gmail.com and hcaawu@gmail.com'

# stats names
REWARD_WEIGHTS_STR = 'Weights'
FEATURE_COUNT_DIFF_STR = 'Feature Count Diff.'
THETA_STR = 'Optimal Weight Vector'
TIME_STR = 'Time'
LEARN_RATE_STR = 'Learning Rate'
LEARNING_DECAY = 0.9

class MultiagentToMMaxEntRewardLearning(ModelLearningAlgorithm):
    """
    An extension of the maximal causal entropy (MaxEnt) algorithm for IRL in [1] to Multi-agent Teams.
    The algorithm is decentralized by applying the conceptL Theory of Mind.
    Other agents act in the learner agent's mind.
    The demonstrated trajectories has the evolving model distribution of the other agents.
    The rest of the process follow MaxEnt IRL
    [1] - Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement
    learning. In AAAI (Vol. 8, pp. 1433-1438).
    """

    def __init__(self,
                 label: str,
                 agent: Agent,
                 reward_vector: LinearRewardVector,
                 processes: Optional[int] = -1,
                 normalize_weights=True,
                 learning_rate: float = 0.01,
                 max_epochs: int = 200,
                 diff_threshold: float = 1e-2,
                 decrease_rate: bool = False,
                 exact=False,
                 num_mc_trajectories=1000,
                 prune_threshold: float = 1e-2,
                 horizon: int = 2,
                 seed: int = 17):
        """
        Creates a new Max Entropy algorithm.
        :param str label: the label associated with this algorithm (might be useful for testing purposes).
        :param Agent agent: the agent whose behavior we want to model (the "expert").
        :param LinearRewardVector reward_vector: the reward vector containing the features whose weights are going to
        be optimized.
        :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
        :param bool normalize_weights: whether to normalize reward weights at each step of the algorithm.
        :param float learning_rate: the gradient descent learning/update rate.
        :param int max_epochs: the maximum number of gradient descent steps.
        :param float diff_threshold: the termination threshold for the weight vector difference.
        :param bool decrease_rate: whether to exponentially decrease the learning rate over time.
        :param bool exact: whether the computation of the distribution over paths should be exact (expand stochastic
        branches) or not, in which case Monte Carlo sample trajectories will be generated to estimate the feature counts.
        :param int num_mc_trajectories: the number of Monte Carlo trajectories to be samples. Works with `exact=False`.
        :param float prune_threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int horizon: the planning horizon used to compute feature counts.
        :param int seed: the seed to initialize the random number generator.
        """
        super().__init__(label, agent.name)

        self.learner_agent = agent
        self.reward_vector = reward_vector
        self.num_features = len(reward_vector)
        self.processes = processes
        self.normalize_weights = normalize_weights
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.diff_threshold = diff_threshold
        self.decrease_rate = decrease_rate
        self.exact = exact
        self.num_mc_trajectories = num_mc_trajectories
        self.prune_threshold = prune_threshold
        self.horizon = horizon
        self.seed = seed

        self.theta = np.ones(self.num_features) / self.num_features

    @staticmethod
    def log_progress(e: int, theta: np.ndarray, diff: float, learning_rate: float, step_time: float):
        with np.printoptions(precision=4, suppress=True):
            print(f'Step {e}: diff={diff:.3f}, θ={theta}, α={learning_rate:.3f}, time={step_time:.2f}s')
            logging.info(f'Step {e}: diff={diff:.3f}, θ={theta}, α={learning_rate:.3f}, time={step_time:.2f}s')

    def learn(self,
              trajectories: List[TeamInfoModelTrajectory],
              data_id: Optional[str] = None,
              verbose: bool = False) -> ModelLearningResult:
        """
        Performs max. entropy model learning by retrieving a PsychSim model containing the reward function approximating
        an expert's behavior as demonstrated through the given trajectories.
        :param list[TeamInfoModelTrajectory] trajectories: a list of team trajectories, each
        containing a list (sequence) of state-team_action-model_dist tuples demonstrated by an "expert" in the task.
        :param str data_id: an (optional) identifier for the data for which model learning was performed.
        :param bool verbose: whether to show information at each timestep during learning.
        :rtype: ModelLearningResult
        :return: the result of the model learning procedure.
        """
        # get empirical feature counts (mean feature path) from trajectories
        feature_func = lambda s: self.reward_vector.get_values(s)
        empirical_fc = expected_feature_counts(trajectories, feature_func)
        if verbose:
            logging.info(f'Empirical Feature Counts {empirical_fc}')
        # estimates information from given trajectories (considered homogenous)
        old_rationality = self.learner_agent.getAttribute('rationality', model=self.learner_agent.get_true_model())
        self.learner_agent.setAttribute('rationality', 1.)
        # traj_len = len(trajectories[0])

        # 1 - initiates reward weights (uniform)
        self.theta = np.ones(self.num_features) / self.num_features

        # 2 - perform gradient descent to optimize reward weights
        diff = np.float('inf')
        e = 0
        step_time = 0
        learning_rate = self.learning_rate
        diffs = [1.] if self.normalize_weights else []
        thetas = [self.theta]
        times = []
        rates = []

        while diff > self.diff_threshold and e < self.max_epochs:
            if verbose:
                self.log_progress(e, self.theta, diff, learning_rate, step_time)

            start = timer()

            # update learning rate
            learning_rate = self.learning_rate
            if self.decrease_rate:
                learning_rate *= np.power(LEARNING_DECAY, e)

            self.reward_vector.set_rewards(self.learner_agent, self.theta)

            # gets expected feature counts (mean feature path)
            # by computing the efc using a MaxEnt stochastic policy given the current reward
            # MaxEnt uses rational agent and we need the distribution over actions if exact

            expected_fc = estimate_feature_counts_with_inference(self.learner_agent, trajectories,
                                                                 self.num_mc_trajectories, feature_func,
                                                                 select=True, horizon=self.horizon,
                                                                 selection='distribution',
                                                                 threshold=self.prune_threshold,
                                                                 processes=self.processes,
                                                                 seed=self.seed, verbose=False, use_tqdm=True)
            if verbose:
                logging.info(f'Estimated Feature Counts {expected_fc}')

            # gradient descent step, update reward weights
            grad = empirical_fc - expected_fc
            new_theta = self.theta + learning_rate * grad
            if self.normalize_weights:
                new_theta /= np.linalg.norm(new_theta, 1)

            step_time = timer() - start

            # registers stats
            diff = np.linalg.norm(new_theta - self.theta)
            diffs.append(diff)
            self.theta = new_theta
            thetas.append(self.theta)
            times.append(step_time)
            rates.append(learning_rate)
            e += 1

        if verbose:
            self.log_progress(e, self.theta, diff, learning_rate, step_time)
            logging.info('Finished, total time: {:.2f} secs.'.format(sum(times)))
        self.learner_agent.setAttribute('rationality', old_rationality)

        # returns stats dictionary
        return ModelLearningResult(data_id, trajectories, {
            FEATURE_COUNT_DIFF_STR: np.array([diffs]),
            REWARD_WEIGHTS_STR: np.array(thetas).T,
            THETA_STR: self.theta,
            TIME_STR: np.array([times]),
            LEARN_RATE_STR: np.array([rates])
        })

    def save_results(self, result: ModelLearningResult, output_dir: str, img_format: str):
        """
        Saves the several results of a run of the algorithm to the given directory.
        :param ModelLearningResult result: the results of the algorithm run.
        :param str output_dir: the path to the directory in which to save the results.
        :param str img_format: the format of the images to be saved.
        :return:
        """
        stats = result.stats
        np.savetxt(os.path.join(output_dir, 'learner-theta.csv'), stats[THETA_STR].reshape(1, -1), '%s', ',',
                   header=','.join(self.reward_vector.names), comments='')

        plot_evolution(stats[FEATURE_COUNT_DIFF_STR], ['diff'], 'Reward Param. Diff. Evolution', None,
                       os.path.join(output_dir, f'evo-rwd-weights-diff.{img_format}'), 'Epoch',
                       '$\Delta \\theta$')

        plot_evolution(stats[REWARD_WEIGHTS_STR], self.reward_vector.names, 'Reward Parameters Evolution', None,
                       os.path.join(output_dir, f'evo-rwd-weights.{img_format}'), 'Epoch', 'Weight', y_lim=[-0.25, 1])

        plot_evolution(stats[TIME_STR], ['time'], 'Step Time Evolution', None,
                       os.path.join(output_dir, f'evo-time.{img_format}'), 'Epoch', 'Time (secs.)')

        plot_evolution(stats[LEARN_RATE_STR], ['learning rate'], 'Learning Rate Evolution', None,
                       os.path.join(output_dir, f'learning-rate.{img_format}'), 'Epoch', 'α')
