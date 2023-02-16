import logging
import os
import numpy as np
from psychsim.reward import maximizeFeature
from psychsim.world import World
from model_learning.planning import get_policy, get_action_values
from model_learning.environments.gridworld import GridWorld
from model_learning.util.io import create_clear_dir
from model_learning.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Collects some trajectories in the normal gridworld with a reward function that tries to maximize ' \
                  'both x and y coordinates of the agent\'s cell location.' \
                  'Plots the trajectories and the reward and value functions.'

ENV_SIZE = 10

AGENT_NAME = 'Agent'
HORIZON = 3
RATIONALITY = 1 / 0.1  # inverse temperature
ACTION_SEL = 'distribution'  # stochastic over all actions
PRUNE_THRESHOLD = 1e-2
SELECT = True

NUM_TRAJECTORIES = 5  # 20
TRAJ_LENGTH = 10  # 15

OUTPUT_DIR = 'output/examples/collect-trajectories'
SEED = 0
PROCESSES = 1
VERBOSE = True
IMG_FORMAT = 'pdf'

if __name__ == '__main__':
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, 'collect.log'))

    # create world and agent
    world = World()
    agent = world.addAgent(AGENT_NAME)
    # agent.setAttribute('selection', SELECTION)
    # agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('rationality', RATIONALITY)

    # create grid-world and add world dynamics to agent
    env = GridWorld(world, ENV_SIZE, ENV_SIZE)
    env.add_agent_dynamics(agent)
    env.plot(os.path.join(OUTPUT_DIR, 'env.png'))

    # set reward function (maximize xy location, ie always move top/right)
    x, y = env.get_location_features(agent)
    agent.setReward(maximizeFeature(x, agent.name), 1.)
    agent.setReward(maximizeFeature(y, agent.name), 1.)

    world.setOrder([{agent.name}])

    # generate trajectories using agent's policy
    logging.info('Generating trajectories...')
    trajectories = env.generate_trajectories(NUM_TRAJECTORIES, TRAJ_LENGTH, agent, select=SELECT,
                                             horizon=HORIZON, selection=ACTION_SEL, threshold=PRUNE_THRESHOLD,
                                             processes=PROCESSES, seed=SEED, verbose=VERBOSE)
    env.log_trajectories(trajectories, agent)
    env.plot_trajectories(trajectories, agent, os.path.join(OUTPUT_DIR, f'trajectories.{IMG_FORMAT}'))

    # gets policy and value
    logging.info('Computing policy...')
    states = env.get_all_states(agent)
    pi = get_policy(agent, states, selection='distribution', threshold=PRUNE_THRESHOLD)
    pi = np.array([[dist[a] if a in dist.domain() else 0. for a in env.agent_actions[agent.name]] for dist in pi])

    logging.info('Computing value function...')
    q = get_action_values(agent, states, env.agent_actions[agent.name], horizon=HORIZON, processes=PROCESSES)
    v = np.max(q, axis=1)  # get state value
    env.plot_policy(pi, v, os.path.join(OUTPUT_DIR, f'policy.{IMG_FORMAT}'))

    # gets rewards
    logging.info('Computing rewards...')
    r = np.array([agent.reward(state) for state in states])
    env.plot_func(r, os.path.join(OUTPUT_DIR, f'reward.{IMG_FORMAT}'), 'Rewards')

    logging.info('\nDone!')
