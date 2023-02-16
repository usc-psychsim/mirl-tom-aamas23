import copy
import numpy as np
from typing import Optional, Literal, List
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.helper_functions import get_true_model_name
from psychsim.probability import Distribution
from psychsim.pwl import modelKey
from model_learning import State
from model_learning.util.mp import run_parallel

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def get_state_policy(agent: Agent,
                     state: State,
                     model: Optional[str] = None,
                     horizon: Optional[int] = None,
                     selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                     threshold: Optional[float] = None) -> Distribution:
    """
    Gets an agent's policy (action selection distribution) for the given state.
    :param Agent agent: the agent for which to calculate the policy.
    :param State state: the state for which to get the agent's policy.
    :param str model: the name of the agent's model used for decision-making. `None` corresponds to the true model.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :rtype: Distribution
    :return: an action distribution for the given state.
    """
    # sets the agent model and performs a fake step to get the agent's decision
    if model is not None:
        state = copy.deepcopy(state)
        agent.world.setFeature(modelKey(agent.name), model, state)
    # p = agent.world.step(state=state, real=True, updateBeliefs=False,
    #                      horizon=horizon, tiebreak=selection, threshold=threshold)
    decision = agent.decide(state, horizon=horizon, selection=selection)
    return decision[agent.world.getFeature(modelKey(agent.name), state=state, unique=True)]['action']
    # return agent.world.getAction(agent.name, agent.world.state)


def get_policy(agent: Agent,
               states: List[State],
               model: Optional[str] = None,
               horizon: Optional[int] = None,
               selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
               threshold: Optional[float] = None,
               processes: int = -1,
               use_tqdm: bool = True) -> List[Distribution]:
    """
    Gets an agent's policy (action selection) for the given states.
    :param Agent agent: the agent for which to calculate the policy.
    :param list[State] states: the list of states for which to get the agent's policy.
    :param str or None model: the name of the agent's model used for decision-making. `None` corresponds to the true model.
    :param int horizon: the agent's planning horizon.
    :param str selection: the action selection criterion, to untie equal-valued actions.
    :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: list[Distribution]
    :return: a list containing an action (or distribution) for each given state.
    """
    args = [(agent, s, model, horizon, selection, threshold) for s in states]
    return run_parallel(get_state_policy, args, processes=processes, use_tqdm=use_tqdm)


def get_state_action_value(agent: Agent,
                           state: State,
                           action: ActionSet,
                           model: Optional[str] = None,
                           horizon: Optional[int] = None) -> float:
    """
    Gets the value (i.e., Q-function) attributed by the agent to the given state-action pair.
    :param Agent agent: the agent for which to calculate the values.
    :param State state:
    :param ActionSet action: the action for which we want to know the value.
    :param str model: the name of the agent's model used for decision-making. `None` corresponds to the true model.
    :param int horizon: the agent's planning horizon.
    :rtype: float
    :return: the value (cumulative discounted reward) attributed by the agent to the given state-action pair.
    """
    return agent.value(state, action, model if model is not None else get_true_model_name(agent), horizon)['__EV__']


def get_action_values(agent: Agent,
                      states: List[State],
                      actions: List[ActionSet],
                      model: Optional[str] = None,
                      horizon: Optional[int] = None,
                      processes: int = -1,
                      use_tqdm: bool = True) -> np.ndarray:
    """
    Gets the values (i.e., Q-function) attributed by the agent to the given states and each action.
    :param Agent agent: the agent for which to calculate the values.
    :param list[State] states: a list of states for which to calculate the value attributed by the agent.
    :param list[ActionSet] actions: a list of actions for which to calculate the value for each state.
    :param str model: the name of the agent's model used for decision-making. `None` corresponds to the true model.
    :param int horizon: the agent's planning horizon.
    :param int processes: number of processes to use. Follows `joblib` convention.
    :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
    :rtype: np.ndarray
    :return: an array of shape (n_states, n_actions) with the values state and action attributed by the agent.
    """
    # gets state-action values for all pairs
    args = [(agent, state, action, model, horizon) for state in states for action in actions]
    values = run_parallel(get_state_action_value, args, processes=processes, use_tqdm=use_tqdm)
    return np.array(values).reshape((len(states), len(actions)))  # shape: (n_states, n_actions)
