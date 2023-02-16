import logging
import os
import numpy as np
from psychsim.world import World
from model_learning.features import estimate_feature_counts
from model_learning.features.objectworld import ObjectsRewardVector
from model_learning.environments.objects_gridworld import ObjectsGridWorld
from model_learning.util.io import create_clear_dir
from model_learning.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = '.'

ENV_SIZE = 10
NUM_OBJECTS = 25
NUM_COLORS = 5
OUTER_COLORS = True
INNER_COLORS = False

AGENT_NAME = 'Agent'
HORIZON = 3
RATIONALITY = 1
# THETA = [0, 0, 0, 0, 0]
THETA = [0.5, -0.4, 0.1, 0., 0.]
PRUNE_THRESHOLD = 1e-4

TRAJ_LENGTH = 10
INIT_X = 0  # 5
INIT_Y = 0  # 5

MC_TRAJECTORIES = 200

OUTPUT_DIR = 'output/examples/compute-feature-counts'
SEED = 123
VERBOSE = True
PROCESSES = -1
IMG_FORMAT = 'pdf'

if __name__ == '__main__':
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, 'collect.log'), level=logging.INFO)

    logging.info('===============================================')

    # create world and agent
    world = World()
    agent = world.addAgent(AGENT_NAME)
    agent.setAttribute('rationality', RATIONALITY)

    # create grid-world and add world dynamics to agent
    env = ObjectsGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_OBJECTS, NUM_COLORS,
                           single_color=OUTER_COLORS ^ INNER_COLORS, seed=SEED)
    env.plot(os.path.join(OUTPUT_DIR, f'env.{IMG_FORMAT}'))

    env.add_agent_dynamics(agent)
    feat_matrix = env.get_location_feature_matrix(OUTER_COLORS, INNER_COLORS)
    rwd_vector = ObjectsRewardVector(env, agent, feat_matrix, OUTER_COLORS, INNER_COLORS)
    rwd_vector.set_rewards(agent, np.array(THETA))

    logging.info(f'Set reward weights to agent: {THETA}')
    logging.info(f'Reward function:\n{agent.getReward()[agent.get_true_model()]}')

    world.setOrder([{agent.name}])

    logging.info('===============================================')
    mean_fc = np.mean(feat_matrix, axis=(0, 1))
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Feature distribution: {mean_fc}')

    # get initial state
    x, y = env.get_location_features(agent)
    world.setFeature(x, INIT_X)
    world.setFeature(y, INIT_Y)
    init_state = world.state

    feature_func = lambda s: rwd_vector.get_values(s)  # to extract the features from the state

    logging.info('===============================================')
    logging.info('Computing expected feature counts using exact distribution...')
    # computes a single "trajectory distribution" using stochastic policy
    efc_dist = estimate_feature_counts(agent, [init_state], TRAJ_LENGTH, feature_func, exact=True,
                                       horizon=HORIZON, threshold=PRUNE_THRESHOLD,
                                       seed=SEED, verbose=VERBOSE)

    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Expected feature counts by computing "trajectory distribution": {efc_dist}')

    logging.info('===============================================')
    logging.info('Computing expected feature counts using Monte Carlo approach to approximate distribution...')
    efc_mc = estimate_feature_counts(agent, [init_state], TRAJ_LENGTH, feature_func, exact=False,
                                     num_mc_trajectories=MC_TRAJECTORIES,
                                     horizon=HORIZON, threshold=PRUNE_THRESHOLD,
                                     processes=PROCESSES, seed=SEED, verbose=VERBOSE)

    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Expected feature counts via Monte Carlo approximation: {efc_mc}')
        logging.info(f'\nEFC absolute difference: {np.abs(efc_dist - efc_mc)}')
