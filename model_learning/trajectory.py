import copy
import random
import logging
import numpy as np
from timeit import default_timer as timer
from typing import List, Dict, Any, Optional, Literal
from psychsim.agent import Agent
from psychsim.world import World
from psychsim.helper_functions import get_random_value
from psychsim.probability import Distribution
from psychsim.pwl import modelKey, turnKey, actionKey, setToConstantMatrix, makeTree
from model_learning import Trajectory, StateActionPair, TeamTrajectory, TeamStateActionPair, \
    TeamStateinfoActionModelTuple, TeamInfoModelTrajectory
from model_learning.util.mp import run_parallel

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

TOP_LEVEL_STR = 'top_level'


def copy_world(world: World) -> World:
    """
    Creates a copy of the given world. This implementation clones the world's state and all agents so that the dynamic
    world elements are "frozen" in time.
    :param World world: the original world to be copied.
    :rtype: World
    :return: a semi-hard copy of the given world.
    """
    new_world = copy.copy(world)
    new_world.state = copy.deepcopy(world.state)
    new_world.agents = copy.copy(new_world.agents)
    for name, agent in world.agents.items():
        # clones agent with exception of world
        agent.world = None
        # TODO tentative
        new_agent = copy.copy(agent)
        new_agent.models = agent.models.copy()
        new_agent.modelList = agent.modelList.copy()
        new_world.agents[name] = new_agent
        new_agent.world = new_world  # assigns cloned world to cloned agent
        agent.world = world  # puts original world back to old agent
    return new_world


def generate_trajectory(agent: Agent,
                        trajectory_length: int,
                        init_feats: Optional[Dict[str, Any]] = None,
                        model: Optional[str] = None,
                        select: Optional[bool] = None,
                        horizon: Optional[int] = None,
                        selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                        threshold: Optional[float] = None,
                        seed: int = 0,
                        verbose: bool = False) -> Trajectory:
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: Trajectory
    :return: a trajectory containing a list of state-action pairs.
    """
    world = copy_world(agent.world)

    # generate or select initial state features
    if init_feats is not None:
        rng = random.Random(seed)
        for feature, init_value in init_feats.items():
            if init_value is None:
                init_value = get_random_value(world, feature, rng)
            elif isinstance(init_value, List):
                init_value = rng.choice(init_value)
            world.setFeature(feature, init_value)

    random.seed(seed)

    # for each step, takes action and registers state-action pairs
    trajectory: Trajectory = []
    total = 0
    prob = 1.
    if model is not None:
        world.setFeature(modelKey(agent.name), model)
    for i in range(trajectory_length):
        start = timer()

        # step the world until it's this agent's turn
        turn = world.getFeature(turnKey(agent.name), unique=True)
        while turn != 0:
            world.step()
            turn = world.getFeature(turnKey(agent.name), unique=True)

        prev_world = copy_world(world)  # keep (possibly stochastic) state and prob before selection
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= world.state.select()

        # steps the world (do not select), gets the agent's action
        world.step(select=False, horizon=horizon, tiebreak=selection, threshold=threshold)
        action = world.getAction(agent.name)
        trajectory.append(StateActionPair(prev_world, action, prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {i} took {step_time:.2f}s (action: {action if len(action) > 1 else action.first()})')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return trajectory


def generate_team_trajectory(team: List[Agent],
                             trajectory_length: int,
                             init_feats: Optional[Dict[str, Any]] = None,
                             model: Optional[str] = None,
                             select: Optional[bool] = None,
                             horizon: Optional[int] = None,
                             selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                             threshold: Optional[float] = None,
                             seed: int = 0,
                             verbose: bool = False) -> TeamTrajectory:
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param List[Agent] team: the team of agents for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: Trajectory
    :return: a trajectory containing a list of state-action pairs.
    """
    world = copy_world(team[0].world)
    if init_feats is not None:
        rng = random.Random(seed)
        for feature, init_value in init_feats.items():
            if init_value is None:
                init_value = get_random_value(world, feature, rng)
            elif isinstance(init_value, List):
                init_value = rng.choice(init_value)
            world.setFeature(feature, init_value)
    random.seed(seed)

    # for each step, takes action and registers state-action pairs
    team_trajectory = []
    # team_trajectory: Dict[str, Trajectory] = {}
    # for agent in team:
    #     team_trajectory[agent.name] = []

    total = 0
    prob = 1.
    if model is not None:
        for agent in team:
            world.setFeature(modelKey(agent.name), model)

    for i in range(trajectory_length):
        start = timer()

        # step the world until it's this agent's turn
        for agent in team:
            turn = world.getFeature(turnKey(agent.name), unique=True)
            while turn != 0:
                world.step()
                turn = world.getFeature(turnKey(agent.name), unique=True)

        prev_world = copy_world(world)  # keep (possibly stochastic) state and prob before selection
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= world.state.select()

        # steps the world (do not select), gets the agent's action
        world.step(select=False, horizon=horizon, tiebreak=selection, threshold=threshold)
        # joint_action TODO
        # append(sapair())
        team_action: Dict[str, Distribution] = {}
        for agent in team:
            action = world.getAction(agent.name)
            team_action[agent.name] = action
        team_trajectory.append(TeamStateActionPair(prev_world, team_action, prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {i} took {step_time:.2f}')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return team_trajectory


def generate_expert_learner_trajectory(expert_team: List[Agent], learner_team: List[Agent],
                                       trajectory_length: int,
                                       init_feats: Optional[Dict[str, Any]] = None,
                                       model: Optional[str] = None,
                                       select: Optional[bool] = None,
                                       horizon: Optional[int] = None,
                                       selection: Optional[
                                           Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                       threshold: Optional[float] = None,
                                       seed: int = 0,
                                       verbose: bool = False) -> TeamTrajectory:
    """
    Generates one fixed-length agent trajectory (state-action pairs) by running the agent in the world.
    :param List[Agent] expert_team: the team of agents that step the world.
    :param List[Agent] learner_team: the team of agents for which to record the actions.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show time information at each timestep during trajectory generation.
    :rtype: Trajectory
    :return: a trajectory containing a list of state-action pairs.
    """
    world = copy_world(expert_team[0].world)
    if init_feats is not None:
        rng = random.Random(seed)
        for feature, init_value in init_feats.items():
            if init_value is None:
                init_value = get_random_value(world, feature, rng)
            elif isinstance(init_value, List):
                init_value = rng.choice(init_value)
            world.setFeature(feature, init_value)
    random.seed(seed)

    # for each step, takes action and registers state-action pairs
    team_trajectory = []
    learner_team_trajectory = []
    # team_trajectory: Dict[str, Trajectory] = {}
    # for agent in team:
    #     team_trajectory[agent.name] = []

    total = 0
    prob = 1.
    if model is not None:
        for agent in expert_team:
            world.setFeature(modelKey(agent.name), model)

    for i in range(trajectory_length):
        start = timer()

        # step the world until it's this agent's turn
        for agent in expert_team:
            turn = world.getFeature(turnKey(agent.name), unique=True)
            while turn != 0:
                world.step()
                turn = world.getFeature(turnKey(agent.name), unique=True)

        prev_world = copy_world(world)  # keep (possibly stochastic) state and prob before selection
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= world.state.select()

        ##
        team_action: Dict[str, Distribution] = {}
        state = copy.deepcopy(world.state)
        # print(state, world.state)
        for agent in learner_team:
            if model is not None:
                agent.world.setFeature(modelKey(agent.name), model, state)
            decision = agent.decide(state, horizon=horizon, selection='distribution')
            action = decision[agent.world.getFeature(modelKey(agent.name), state=state, unique=True)]
            # print(agent.name+'_learner', Distribution({action['action']: 1}))
            # team_action[agent.name + '_learner'] = Distribution({action['action']: 1})
            team_action[agent.name + '_learner'] = action['action']

        # steps the world (do not select), gets the agent's action
        for agent in expert_team:
            if model is not None:
                agent.world.setFeature(modelKey(agent.name), model, state)
            decision = agent.decide(state, horizon=horizon, selection='distribution')
            action = decision[agent.world.getFeature(modelKey(agent.name), state=state, unique=True)]
            # print(agent.name, Distribution({action['action']: 1}))
            # team_action[agent.name] = Distribution({action['action']: 1})
            team_action[agent.name] = action['action']

        world.step(select=False, horizon=horizon, tiebreak=selection, threshold=threshold)

        # for agent in expert_team:
        #     action = world.getAction(agent.name)
        #     print(agent.name, action)
        #     team_action[agent.name] = action

        team_trajectory.append(TeamStateActionPair(prev_world, team_action, prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {i} took {step_time:.2f}')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return team_trajectory


def generate_trajectories(agent: Agent,
                          n_trajectories: int,
                          trajectory_length: int,
                          init_feats: Optional[Dict[str, Any]] = None,
                          model: Optional[str] = None,
                          select: Optional[bool] = None,
                          horizon: Optional[int] = None,
                          selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                          threshold: Optional[float] = None,
                          processes: int = -1,
                          seed: int = 0,
                          verbose: bool = False,
                          use_tqdm: bool = True) -> List[Trajectory]:
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
    :param Agent agent: the agent for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """
    # initial checks
    world = agent.world
    if init_feats is not None:
        for feature in init_feats:
            assert feature in world.variables, f'World does not have feature \'{feature}\'!'

    # generates each trajectory in parallel using a different random seed
    start = timer()
    args = [(agent, trajectory_length, init_feats, model, select, horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[Trajectory] = run_parallel(generate_trajectory, args, processes=processes, use_tqdm=use_tqdm)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories of length {trajectory_length}: '
                     f'{timer() - start:.3f}s')

    return trajectories


def generate_team_trajectories(team: List[Agent],
                               n_trajectories: int,
                               trajectory_length: int,
                               init_feats: Optional[Dict[str, Any]] = None,
                               model: Optional[str] = None,
                               select: Optional[bool] = None,
                               horizon: Optional[int] = None,
                               selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                               threshold: Optional[float] = None,
                               processes: int = -1,
                               seed: int = 0,
                               verbose: bool = False,
                               use_tqdm: bool = True) -> List[TeamTrajectory]:
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
    :param List[Agent] team: the team of agents for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[Trajectory]
    :return: a list of trajectories, each containing a list of state-action pairs.
    """
    # initial checks
    for ag_i, agent in enumerate(team):
        assert agent.world == team[ag_i - 1].world
    world = team[0].world
    for feature in init_feats:
        assert feature in world.variables, f'World does not have feature \'{feature}\'!'

    # generates each trajectory in parallel using a different random seed
    start = timer()
    args = [(team, trajectory_length, init_feats, model, select, horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[TeamTrajectory] = run_parallel(generate_team_trajectory, args, processes=processes,
                                                      use_tqdm=use_tqdm)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories of length {trajectory_length}: '
                     f'{timer() - start:.3f}s')

    return trajectories


def generate_expert_learner_trajectories(expert_team: List[Agent], learner_team: List[Agent],
                                         n_trajectories: int,
                                         trajectory_length: int,
                                         init_feats: Optional[Dict[str, Any]] = None,
                                         model: Optional[str] = None,
                                         select: Optional[bool] = None,
                                         horizon: Optional[int] = None,
                                         selection: Optional[
                                             Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                         threshold: Optional[float] = None,
                                         processes: int = -1,
                                         seed: int = 0,
                                         verbose: bool = False,
                                         use_tqdm: bool = True) -> List[TeamTrajectory]:
    """
    Generates a number of fixed-length agent trajectories (state-action pairs) by running the agent in the world.
    :param List[Agent] expert_team: the team of agents that step the world.
    :param List[Agent] learner_team: the team of agents for which to record the actions.
    :param int n_trajectories: the number of trajectories to be generated.
    :param int trajectory_length: the length of the generated trajectories.
    :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
    trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
    values to choose from, a single value, or `None`, in which case a random value will be picked based on the
    feature's domain.
    :param str model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[TeamTrajectory]
    :return: a list of trajectories, each containing a list of state-expert-learner-action pairs.
    """
    # initial checks
    for ag_i, agent in enumerate(expert_team):
        assert agent.world == expert_team[ag_i - 1].world
    world = expert_team[0].world
    for feature in init_feats:
        assert feature in world.variables, f'World does not have feature \'{feature}\'!'

    # generates each trajectory in parallel using a different random seed
    start = timer()
    args = [(expert_team, learner_team, trajectory_length, init_feats, model,
             select, horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    # trajectories: Dict[str, List[TeamTrajectory]] = {}
    trajectories: List[TeamTrajectory] = run_parallel(generate_expert_learner_trajectory, args, processes=processes,
                                                      use_tqdm=use_tqdm)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories of length {trajectory_length}: '
                     f'{timer() - start:.3f}s')

    return trajectories


def generate_trajectory_with_inference(learner_agent: Agent,
                                       team_trajs: List[TeamInfoModelTrajectory],
                                       traj_i: int,
                                       learner_model: Optional[str] = None,
                                       select: Optional[bool] = None,
                                       horizon: Optional[int] = None,
                                       selection: Optional[
                                           Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                                       threshold: Optional[float] = None,
                                       seed: int = 0,
                                       verbose: bool = False
                                       ) -> TeamInfoModelTrajectory:
    """
    Generates a number of fixed-length agent trajectories with model inference of the other agents.
    :param Agent learner_agent: the agent for which to record the actions
    :param List[TeamInfoModelTrajectory] team_trajs: the recorded trajectories of inference
    :param int traj_i: the index of the recorded trajectory
    :param str learner_model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :rtype: TeamInfoModelTrajectory]
    :return: the trajectory containing a list of state-action-model_distribution tuples.
    """
    random.seed(seed)
    n_step = len(team_trajs[traj_i])
    init_state = team_trajs[traj_i][0].state

    trajectory: TeamInfoModelTrajectory = []
    # reset to initial state and uniform dist of models
    _world = copy_world(learner_agent.world)
    init_state = copy.deepcopy(init_state)
    del init_state[modelKey('observer')]
    _world.state = init_state

    team_agents = list(_world.agents.keys())
    _team = [_world.agents[agent_name] for agent_name in team_agents]  # get new world's agents
    learner_agent = _team[team_agents.index(learner_agent.name)]
    if learner_model is not None:
        _world.setFeature(modelKey(learner_agent.name), learner_model)

    # learner_agent.set_observations()
    _world.setOrder([{_agent.name for _agent in _team}])
    _world.dependency.getEvaluation()

    total = 0
    prob = 1
    for step_i in range(n_step):
        # print('====================')
        # print(f'Step {step_i}:')
        start = timer()

        other_actions = {}
        for ag_i, _agent in enumerate(_team):
            if _agent.name != learner_agent.name:
                true_model = _agent.get_true_model()
                model_names = [name for name in _agent.models.keys() if name != true_model]

                included_features = set(learner_agent.getBelief(model=learner_agent.get_true_model()).keys())
                learner_agent.resetBelief(include=included_features)  # override actual features in belief states

                dist = Distribution({model: team_trajs[traj_i][step_i].model_dist[model] for model in model_names})
                # dist = Distribution({f'{_agent.name}_GroundTruth': 1,
                #                      f'{_agent.name}_Opposite': 0,
                #                      f'{_agent.name}_Random': 0, })
                # dist = Distribution({f'{_agent.name}_Random': 0,
                #                      f'{_agent.name}_Task': 1,
                #                      f'{_agent.name}_Social': 0,})
                if select:
                    dist, model_prob = dist.sample()
                learner_agent.setBelief(modelKey(_agent.name), dist)

                action_dist = []
                model_names = _world.getFeature(modelKey(_agent.name),
                                                state=learner_agent.getBelief(model=learner_agent.get_true_model()))

                for model, model_prob in model_names.items():
                    decision = _agent.decide(selection='random', model=model)
                    action = decision['action']
                    action = _world.value2float(actionKey(_agent.name), action)
                    action_dist.append((setToConstantMatrix(actionKey(_agent.name), action), model_prob))
                other_actions[_agent.name] = makeTree({'distribution': action_dist})

        # step the world until it's this agent's turn
        turn = _world.getFeature(turnKey(learner_agent.name), unique=True)
        while turn != 0:
            _world.step()
            turn = _world.getFeature(turnKey(learner_agent.name), unique=True)

        prev_world = copy_world(_world)
        prev_prob = prob
        if select:
            # select if state is stochastic and update probability of reaching state
            prob *= _world.state.select()

        _world.step(actions=other_actions, select=False, horizon=horizon, tiebreak=selection,
                    threshold=threshold, updateBeliefs=False)

        action = _world.getAction(learner_agent.name)
        trajectory.append(TeamStateinfoActionModelTuple(prev_world.state,
                                                        action,
                                                        team_trajs[traj_i][step_i].model_dist,
                                                        prev_prob))

        step_time = timer() - start
        total += step_time
        if verbose:
            logging.info(f'Step {step_i} took {step_time:.2f}')

    if verbose:
        logging.info(f'Total time: {total:.2f}s')

    return trajectory


def generate_trajectories_with_inference(learner_agent: Agent,
                                         team_trajs: List[TeamInfoModelTrajectory],
                                         traj_i: int,
                                         n_trajectories: int,
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
                                         use_tqdm: bool = True
                                         ) -> List[TeamInfoModelTrajectory]:
    """
    Generates a number of fixed-length agent trajectories with model inference of the other agents.
    :param Agent learner_agent: the agent for which to record the actions
    :param List[TeamInfoModelTrajectory] team_trajs: the recorded trajectories of inference
    :param int traj_i: the index of the recorded trajectory
    :param int n_trajectories: the number of trajectories to be generated.
    :param bool exact: whether to perform exact step of the world
    :param str learner_model: the agent model used to generate the trajectories.
    :param bool select: whether to select from stochastic states after each world step.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param int seed: the seed used to initialize the random number generator.
    :param bool verbose: whether to show information at each timestep during trajectory generation.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[TeamInfoModelTrajectory]
    :return: a list of trajectories, each containing a list of state-action-model_distribution tuples.
    """
    if exact:
        # exact computation, generate single stochastic trajectory (select=False) from initial state
        trajectory_dist = generate_trajectory_with_inference(learner_agent, team_trajs, traj_i,
                                                             learner_model=learner_model, select=True,
                                                             horizon=horizon, selection='distribution',
                                                             threshold=threshold,
                                                             seed=seed, verbose=verbose)
        return [trajectory_dist]

    start = timer()
    args = [(learner_agent, team_trajs, traj_i, learner_model, select,
             horizon, selection, threshold, seed + t, verbose)
            for t in range(n_trajectories)]
    trajectories: List[TeamInfoModelTrajectory] = run_parallel(generate_trajectory_with_inference, args,
                                                               processes=processes, use_tqdm=False)

    if verbose:
        logging.info(f'Total time for generating {n_trajectories} trajectories: '
                     f'{timer() - start:.3f}s')

    return trajectories


def sample_random_sub_trajectories(trajectory: Trajectory,
                                   n_trajectories: int,
                                   trajectory_length: int,
                                   with_replacement: bool = False,
                                   seed: int = 0) -> List[Trajectory]:
    """
    Randomly samples sub-trajectories from a given trajectory.
    :param Trajectory trajectory: the trajectory containing a list of state-action pairs.
    :param int n_trajectories: the number of trajectories to be sampled.
    :param int trajectory_length: the length of the sampled trajectories.
    :param bool with_replacement: whether to allow repeated sub-trajectories to be sampled.
    :param int seed: the seed used to initialize the random number generator.
    :rtype: list[Trajectory]
    :return: a list of sub-trajectories, each containing a list of state-action pairs.
    """
    # check provided trajectory length
    assert with_replacement or len(trajectory) > trajectory_length + n_trajectories - 1, \
        'Trajectory has insufficient length in relation to the requested length and amount of sub-trajectories.'

    # samples randomly
    rng = random.Random(seed)
    idxs = list(range(len(trajectory) - trajectory_length + 1))
    idxs = rng.choices(idxs, k=n_trajectories) if with_replacement else rng.sample(idxs, n_trajectories)
    return [trajectory[idx:idx + trajectory_length] for idx in idxs]


def sample_spread_sub_trajectories(trajectory: Trajectory,
                                   n_trajectories: int,
                                   trajectory_length: int) -> List[Trajectory]:
    """
    Samples sub-trajectories from a given trajectory as spread in time as possible.
    :param Trajectory trajectory: the trajectory containing a list of state-action pairs.
    :param int n_trajectories: the number of trajectories to be sampled.
    :param int trajectory_length: the length of the sampled trajectories.
    :rtype: list[Trajectory]
    :return: a list of sub-trajectories, each containing a list of state-action pairs.
    """
    # check provided trajectory length
    assert len(trajectory) > trajectory_length + n_trajectories - 1, \
        'Trajectory has insufficient length in relation to the requested length and amount of sub-trajectories.'

    idxs = np.asarray(np.arange(0, len(trajectory) - trajectory_length + 1,
                                (len(trajectory) - trajectory_length) / max(1, n_trajectories - 1)), dtype=int)
    return [trajectory[idx:idx + trajectory_length] for idx in idxs]


def log_trajectories(trajectories: List[Trajectory], features: List[str]):
    """
    Prints the given trajectories to the log at the info level.
    :param list[list[tuple[World, Distribution]]] trajectories: the set of trajectories to save, containing
    several sequences of state-action pairs.
    :param list[str] features: the state features to be printed at each step, representing the state of interest.
    """
    if len(trajectories) == 0 or len(trajectories[0]) == 0:
        return

    for i, trajectory in enumerate(trajectories):
        logging.info('-------------------------------------------')
        logging.info(f'Trajectory {i}:')
        for t, sa in enumerate(trajectory):
            action = sa.action
            feat_values = [str(sa.world.getFeature(feat)) for feat in features]
            logging.info(f'{t}:\t({", ".join(feat_values)}) -> {action if len(action) > 1 else action.first()}')
