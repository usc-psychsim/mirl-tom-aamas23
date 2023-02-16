import logging
import os
import numpy as np
from psychsim.world import World
from model_learning.environments.objects_gridworld import ObjectsGridWorld
from model_learning.planning import get_policy, get_action_values
from model_learning.util.io import create_clear_dir
from model_learning.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

ENV_SIZE = 10
NUM_OBJECTS = 25
NUM_COLORS = 5

AGENT_NAME = 'Agent'
HORIZON = 3
RATIONALITY = 1 / 0.05  # inverse temperature
ACTION_SEL = 'random'  # 'distribution'  # stochastic over all actions
PRUNE_THRESHOLD = 1e-2
SELECT = True

NUM_TRAJECTORIES = 5  # 10
TRAJ_LENGTH = 10  # 20

OUTPUT_DIR = 'output/examples/collect-traj-obj-world'
SEED = 0
PROCESSES = -1
VERBOSE = True
IMG_FORMAT = 'pdf'

if __name__ == '__main__':
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, 'collect.log'), level=logging.INFO)

    # create world and agent
    world = World()
    agent = world.addAgent(AGENT_NAME)
    # agent.setAttribute('selection', SELECTION)
    # agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('rationality', RATIONALITY)

    # create grid-world and add world dynamics to agent
    env = ObjectsGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_OBJECTS, NUM_COLORS, seed=SEED)
    env.add_agent_dynamics(agent)
    env.plot(os.path.join(OUTPUT_DIR, f'env.{IMG_FORMAT}'))

    # set reward function (love object 0, hate object 1, object 2 is ok, others are distractions)
    weights = np.array([0.4, -0.4, 0.1, 0., 0.])
    env.set_linear_color_reward(agent, weights)
    logging.info(f'Set reward function to agent:\n{agent.getReward()[agent.get_true_model()]}')

    world.setOrder([{agent.name}])

    logging.info('Computing color feature maps...')
    feat_matrix = env.get_location_feature_matrix(outer=True, inner=False)
    for c in range(env.num_colors):
        c_f = feat_matrix[..., c].T  # need to transpose XY
        env.plot_func(c_f, os.path.join(OUTPUT_DIR, f'color_{c}_feat.{IMG_FORMAT}'), f'Color {c} Features')

    feat_matrix = env.get_distance_feature_matrix(outer=False, inner=True)
    for c in range(env.num_colors):
        c_f = feat_matrix[..., c].T  # need to transpose XY
        env.plot_func(c_f, os.path.join(OUTPUT_DIR, f'color_{c}_dist_feat.{IMG_FORMAT}'),
                      f'Color {c} Distance Features')

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
    pi = get_policy(agent, states, selection='distribution', threshold=PRUNE_THRESHOLD, processes=PROCESSES)
    pi = np.array([[dist[a] if a in dist.domain() else 0. for a in env.agent_actions[agent.name]] for dist in pi])

    logging.info('Computing value function...')
    q = get_action_values(agent, states, env.agent_actions[agent.name], horizon=HORIZON, processes=PROCESSES)
    v = np.max(q, axis=1)  # get state value
    env.plot_policy(pi, v, os.path.join(OUTPUT_DIR, f'policy.{IMG_FORMAT}'))

    # gets rewards
    logging.info('Computing rewards...')
    r = np.array([agent.reward(state) for state in states])
    env.plot_func(r, os.path.join(OUTPUT_DIR, f'reward.{IMG_FORMAT}'), 'Rewards')
