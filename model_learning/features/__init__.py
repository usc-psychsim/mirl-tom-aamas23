import itertools as it
import numpy as np
from typing import List, Callable, Optional, Literal, Union
from model_learning.features.linear import LinearRewardVector
from model_learning.util.mp import run_parallel
from psychsim.agent import Agent
from model_learning import Trajectory, State, TeamInfoModelTrajectory
from model_learning.trajectory import generate_trajectory, generate_trajectories, copy_world, \
    generate_trajectories_with_inference, generate_trajectory_with_inference

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def expected_feature_counts(trajectories: Union[List[Trajectory], List[TeamInfoModelTrajectory]],
                            feature_func: Callable[[State], np.ndarray]) -> np.ndarray:
    """
    Computes the expected (mean over paths) feature counts, i.e., the sum of the feature values for each state along
    a trajectory, averaged across all given trajectories and weighted according to the probability of each trajectory.
    :param list[Trajectory] trajectories: a list of trajectories, each containing a sequence of state-action pairs.
    :param Callable feature_func: the function to extract the features out of each state.
    :rtype: np.ndarray
    :return: the mean counts for each feature over all trajectories.
    """
    # gets feature counts for each timestep of each trajectory
    t_fcs = []
    t_probs = []
    for trajectory in trajectories:
        fcs = []
        probs = []
        for sap in trajectory:
            # gets feature values at this state weighted by its probability, shape: (num_features, )
            if hasattr(sap, 'world'):
                fcs.append(feature_func(sap.world.state) * sap.prob)
            else:
                fcs.append(feature_func(sap.state) * sap.prob)
            probs.append(sap.prob)
        t_probs.append(np.array(probs).reshape(-1, 1))  # get probs during trajectory, shape: (timesteps, 1)
        t_fcs.append(np.array(fcs))  # shape: (timesteps, num_features)

    t_probs = np.array(t_probs)  # shape: (num_traj, timesteps, 1)
    prob_weights = np.sum(t_probs, axis=0)  # shape: (timesteps, 1)
    t_fcs = np.array(t_fcs)  # shape: (num_traj, timesteps, num_features)
    fcs_weighted = np.sum(t_fcs, axis=0) / prob_weights  # shape: (timesteps, num_features)
    return np.sum(fcs_weighted, axis=0)  # get weighted average of feature counts/sums, shape: (num_features, )


def estimate_feature_counts_with_inference(learner_agent: Agent,
                                           team_trajs: List[TeamInfoModelTrajectory],
                                           n_trajectories: int,
                                           feature_func: Callable[[State], np.ndarray],
                                           exact: bool = False,
                                           learner_model: Optional[str] = None,
                                           select: Optional[bool] = None,
                                           horizon: Optional[int] = None,
                                           selection: Optional[
                                               Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                           threshold: Optional[float] = None,
                                           processes: int = -1,
                                           seed: int = 0,
                                           verbose: bool = False,
                                           use_tqdm: bool = True) -> np.ndarray:

    args = []
    for traj_i, team_traj in enumerate(team_trajs):
        args.append((learner_agent, team_trajs, traj_i, n_trajectories, exact, learner_model,
                     select, horizon, selection, threshold, processes, seed + traj_i, verbose))
    trajectories = run_parallel(generate_trajectories_with_inference, args, processes=processes, use_tqdm=use_tqdm)
    trajectories = list(it.chain(*trajectories))
    # print(trajectories)
    return expected_feature_counts(trajectories, feature_func)


def estimate_feature_counts(agent: Agent,
                            initial_states: List[State],
                            trajectory_length: int,
                            feature_func: Callable[[State], np.ndarray],
                            exact: bool = False,
                            num_mc_trajectories: int = 100,
                            model: Optional[str] = None,
                            horizon: Optional[int] = None,
                            threshold: Optional[float] = None,
                            processes: Optional[int] = -1,
                            seed: int = 0,
                            verbose: bool = False,
                            use_tqdm: bool = True) -> np.ndarray:
    """
    Estimates the expected feature counts by generating trajectories from the given initial states and then computing
    the average feature counts per path.
    :param Agent agent: the agent for which to compute the expected feature counts.
    :param list[State] initial_states: the list of initial states from which trajectories should be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param Callable feature_func: the function to extract the features out of each state.
    :param bool exact: whether the computation of the distribution over paths should be exact (expand stochastic
    branches) or not, in which case Monte Carlo sample trajectories will be generated to estimate the feature counts.
    :param int num_mc_trajectories: the number of Monte Carlo trajectories to be samples. Works with `exact=False`.
    :param str model: the agent model used to generate the trajectories.
    :param int horizon: the agent's planning horizon.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: np.ndarray
    :return: the estimated expected feature counts.
    """

    args = []
    for t, initial_state in enumerate(initial_states):
        args.append((agent, initial_state, trajectory_length, exact, num_mc_trajectories, model, horizon,
                     threshold, processes, seed + t, verbose))
    trajectories = run_parallel(_generate_trajectories, args, processes=processes, use_tqdm=use_tqdm)
    trajectories = list(it.chain(*trajectories))

    # return expected feature counts over all generated trajectories
    return expected_feature_counts(trajectories, feature_func)


def _generate_trajectories(
        agent: Agent, initial_state: State, trajectory_length: int, exact: bool,
        num_mc_trajectories: int, model: Optional[str], horizon: Optional[int],
        threshold: Optional[float], processes: Optional[int], seed: int, verbose: bool) -> List[Trajectory]:
    # make copy of world and set initial state
    world = copy_world(agent.world)
    _agent = world.agents[agent.name]
    world.state = initial_state

    if exact:
        # exact computation, generate single stochastic trajectory (select=False) from initial state
        trajectory_dist = generate_trajectory(_agent, trajectory_length, model=model, select=False,
                                              horizon=horizon, selection='distribution', threshold=threshold,
                                              seed=seed, verbose=verbose)
        return [trajectory_dist]

    # Monte Carlo approximation, generate N single-path trajectories (select=True) from initial state
    trajectories_mc = generate_trajectories(_agent, num_mc_trajectories, trajectory_length,
                                            model=model, select=True,
                                            horizon=horizon, selection='distribution', threshold=threshold,
                                            processes=processes, seed=seed, verbose=verbose,
                                            use_tqdm=False)
    return trajectories_mc
