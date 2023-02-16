import logging
import numpy as np
from psychsim.probability import Distribution
from psychsim.world import World
from model_learning.evaluation.metrics import evaluate_internal
from model_learning.features.linear import LinearRewardVector
from model_learning.planning import get_policy
from model_learning.trajectory import sample_random_sub_trajectories

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def cross_evaluation(trajectories, agent_names, rwd_vectors, rwd_weights,
                     rationality=None, horizon=None, threshold=None, processes=None,
                     num_states=None, seed=0, invalid_val=-1.):
    """
    Performs cross internal evaluation by testing a set of agents under different linear reward weights.
    These can represent different hypothesis, different notional models, the centers of different clusters, etc.
    :param list[list[tuple[World, Distribution]]] trajectories: the set of trajectories containing the experts' behavior
    against which we will compare the behavior resulting from the different reward weight vectors, containing several
    sequences of state-action pairs.
    :param list[str] agent_names: the names of the expert agents in each trajectory used for evaluation.
    :param list[LinearRewardVector] rwd_vectors: the reward functions for each agent.
    :param list[np.ndarray] rwd_weights: the reward weight vectors that we want to compare against the expert's behavior.
    :param float rationality: the rationality of agents when computing their policy to compare against the experts'.
    :param int horizon: the agent's planning horizon.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
    :param int num_states: the number of states in the trajectory to be evaluated. `None` indicates all states.
    :param int seed: the seed used to initialize the random number generator for state sampling.
    :param float invalid_val: the value set to non-compared pairs, used to initialize the confusion matrix.
    :rtype: dict[str, np.ndarray]
    :return: a dictionary containing, for each internal evaluation metric, a matrix of size (num_rwd_weights,
    num_trajectories) with the evaluation comparing each expert's policy (column) against each reward weight vector (row).
    """
    assert len(trajectories) == len(agent_names) == len(rwd_vectors), \
        'Different number of trajectories, agent names or reward vectors provided!'

    eval_matrix = {}
    for i, trajectory in enumerate(trajectories):
        agent = trajectory[-1][0].agents[agent_names[i]]
        agent.setAttribute('rationality', rationality)

        # sample states from trajectories
        worlds = trajectory if num_states is None else \
            [st[0] for st in sample_random_sub_trajectories(trajectory, num_states, 1, seed=seed)]

        # expert's observed "policy"
        expert_states = [w.state for w, _ in worlds]
        expert_pi = [a for _, a in worlds]

        # compute agent's policy under each reward function
        for j, rwd_weight in enumerate(rwd_weights):

            rwd_vectors[i].set_rewards(agent, rwd_weights[j])
            with np.printoptions(precision=2, suppress=True):
                logging.info('Computing policy for agent {} with reward {} for {} states...'.format(
                    agent.name, rwd_weights[j], len(expert_states)))
            agent_pi = get_policy(agent, expert_states, None, horizon, 'distribution', threshold, processes)

            # gets internal performance metrics between policies and stores in matrix
            metrics = evaluate_internal(expert_pi, agent_pi)
            for metric_name, metric in metrics.items():
                if metric_name not in eval_matrix:
                    eval_matrix[metric_name] = np.full((len(rwd_weights),len(trajectories)), invalid_val)
                eval_matrix[metric_name][j, i] = metric

    return eval_matrix
