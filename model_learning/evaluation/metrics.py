import numpy as np
from typing import List, Dict
from psychsim.probability import Distribution
from model_learning.util.math import jensen_shannon_divergence

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

POLICY_MISMATCH_STR = 'Policy Mismatch'
POLICY_DIVERGENCE_STR = 'Policy Divergence'


def evaluate_internal(expert_pi: List[Distribution], learner_pi: List[Distribution]) -> Dict[str, float]:
    """
    Utility method that compares two policies using internal metrics, i.e., that do not rely on knowing the
    "ground-truth" expert policy or reward function. Useful for analyzing human behavior data.
    Note: the policies are lists where elements are distributions over actions for each state considered. Policies
    should have the same size and it is assumed that each pair of elements.
    :param list[Distribution] expert_pi: the expert's policy (or approximation) for some set of states.
    :param list[Distribution] learner_pi: the learner's policy for the same set of states.
    :rtype: dict[str, float]
    :return: a dictionary containing several internal evaluation metrics.
    """
    return {
        POLICY_MISMATCH_STR: policy_mismatch_prob(expert_pi, learner_pi),
        POLICY_DIVERGENCE_STR: policy_divergence(expert_pi, learner_pi)
    }


def policy_mismatch_prob(policy1: List[Distribution], policy2: List[Distribution]) -> float:
    """
    Compares the given policies by measuring the amount of discrepancy between them. This corresponds to the mean,
    over all states, of the probability that the policies will take different actions.
    Note: the policies are lists where elements are distributions over actions for each state considered. Policies
    should have the same size and it is assumed that each pair of elements.
    corresponds to the policy in the same state.
    :param list[Distribution] policy1: the first policy.
    :param list[Distribution] policy2: the second policy.
    :rtype float:
    :return: the mean probability in `[0, 1]`, of the policies choosing different actions.
    """
    assert len(policy1) == len(policy2), f'Given policies have different lengths: {len(policy1)}!={len(policy2)}'

    # computes prob. of different actions for all states
    prob = 0.
    for s in range(len(policy1)):
        a1 = policy1[s]
        a2 = policy2[s]
        joint = sum(a1.get(a) * a2.get(a) for a in set(list(a1.domain()) + list(a2.domain())))
        prob += 1. - joint
    return prob / len(policy1)


def policy_divergence(policy1: List[Distribution], policy2: List[Distribution]) -> float:
    """
    Compares the given policies by measuring the amount of discrepancy between them in terms of probability divergence.
    This is measured by the mean, over all states, of the Jensen-Shannon divergence (JSD) between the policies'
    probability distribution over actions.
    Note: the policies are lists where elements are distributions over actions for each state considered. Policies
    should have the same size and it is assumed that each pair of elements.
    corresponds to the policy in the same state.
    :param list[Distribution] policy1: the first policy.
    :param list[Distribution] policy2: the second policy.
    :rtype float:
    :return: the mean probability in `[0, 1]`, of the policies choosing different actions.
    """
    assert len(policy1) == len(policy2), f'Given policies have different lengths: {len(policy1)}!={len(policy2)}'

    # computes JSD divergence for all states
    div = 0.
    for s in range(len(policy1)):
        # gets stochastic distributions over actions
        a1 = policy1[s]
        a2 = policy2[s]

        # gets JSD between the two
        actions = list(set(list(a1.domain()) + list(a2.domain())))
        a1 = np.array([a1.get(a) for a in actions])  # shape (num_actions, )
        a2 = np.array([a2.get(a) for a in actions])  # shape (num_actions, )
        dists = np.array([a1, a2])+1e-5  # shape (2, num_actions)
        div += jensen_shannon_divergence(dists)

    return div / len(policy1)

# TODO
# def evd(env, expert_r, learner_r, trajectories, stochastic=False):
#     """
#     Computes the expected value difference (EVD) between the expert and learner's policies.
#     Measures how suboptimal the learned policy is under the true reward. We find the optimal policy under the learned
#     reward, measure its expected sum of discounted rewards under the true (expert) reward function, and subtract this
#     quantity from the expected sum of discounted rewards under the true (expert) policy.
#     :param Environment env: the MDP/environment.
#     :param np.ndarray expert_r: the true (expert) reward function shaped (n_states, 1).
#     :param np.ndarray learner_r: the reward function derived via IRL shaped (n_states, 1).
#     :param np.ndarray trajectories: the expert's demonstrated trajectories, shaped (n_trajectories, traj_length).
#     :param bool stochastic: whether the MDP's policies should be stochastic for the calculation of the EVD.
#     :rtype: (float, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
#     :return: a tuple containing: the expected value difference (EVD) between the expert's and learner's policy,
#     the expert's computed policy, the expert's computed value function, the learner's computed policy, the learner's
#     computed value function.
#     """
#     # computes expert value function given expert/true reward
#     expert_q = value_iteration(env.transition_probability, expert_r, env.discount)
#     expert_policy = softmax_policy(expert_q, stochastic)
#     expert_v = np.max(expert_q, axis=1)
#
#     # computes learner value function given the learned/optimized reward under the expert's reward
#     learner_q = value_iteration(env.transition_probability, learner_r, env.discount)
#     learner_policy = softmax_policy(learner_q, stochastic)
#     learner_v = np.max(learner_q, axis=1)
#
#     # initial state probabilities derived from demonstrations
#     p_start_state = np.bincount(trajectories[:, 0, 0], minlength=env.n_states) / trajectories.shape[0]
#
#     # computes difference in expected value
#     evd = np.abs(expert_v.dot(p_start_state) - learner_v.dot(p_start_state))
#
#     return evd, expert_policy, expert_v, learner_policy, learner_v
