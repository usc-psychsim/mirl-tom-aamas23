import logging
import os
import numpy as np
from psychsim.world import World
from model_learning.algorithms.max_entropy import MaxEntRewardLearning, THETA_STR
from model_learning.features.objectworld import ObjectsRewardVector
from model_learning.environments.objects_gridworld import ObjectsGridWorld
from model_learning.evaluation.metrics import policy_mismatch_prob, policy_divergence
from model_learning.planning import get_policy, get_action_values
from model_learning.util.logging import change_log_handler
from model_learning.util.io import create_clear_dir

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Performs IRL (reward mode learning) in the Objects World using MaxEnt IRL.'

# env params
ENV_SIZE = 10
NUM_OBJECTS = 25
NUM_COLORS = 5
OUTER_COLORS = True
INNER_COLORS = False
ENV_SEED = 17

# expert params
EXPERT_NAME = 'Agent'
EXPERT_THETA = [0.5, -0.4, 0.1, 0., 0.]
EXPERT_RATIONALITY = 1 / 0.1  # inverse temperature
EXPERT_ACT_SELECTION = 'random'
EXPERT_SEED = 17
NUM_TRAJECTORIES = 2  # 8 # 20
TRAJ_LENGTH = 10  # 10 # 15

# learning params
NORM_THETA = True
LEARNING_RATE = 1e-2  # 0.01
MAX_EPOCHS = 200
THRESHOLD = 1e-3
DECREASE_RATE = True
EXACT = False
NUM_MC_TRAJECTORIES = 200 #200
LEARNING_SEED = 17

# common params
HORIZON = 3
PRUNE_THRESHOLD = 1e-2

OUTPUT_DIR = 'output/examples/object-world-irl'
IMG_FORMAT = 'pdf'
PROCESSES = -1
VERBOSE = True

if __name__ == '__main__':
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, 'irl.log'), logging.INFO)

    # create world and objects environment
    world = World()
    env = ObjectsGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_OBJECTS, NUM_COLORS, seed=ENV_SEED)
    env.plot(os.path.join(OUTPUT_DIR, f'env.{IMG_FORMAT}'))

    # create expert and add world dynamics and reward function
    logging.info('=================================')
    agent = world.addAgent(EXPERT_NAME)
    agent.setAttribute('selection', EXPERT_ACT_SELECTION)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('rationality', EXPERT_RATIONALITY)
    env.add_agent_dynamics(agent)
    env.set_linear_color_reward(agent, weights_outer=np.array(EXPERT_THETA))
    logging.info(f'Set reward weights to expert: {EXPERT_THETA}')
    logging.info(f'Reward function:\n{agent.getReward()[agent.get_true_model()]}')

    world.setOrder([{agent.name}])

    # gets all env states
    states = env.get_all_states(agent)

    # gets policy and value
    logging.info('=================================')
    logging.info('Computing expert policy & value function...')
    expert_pi = get_policy(agent, states, selection='distribution', threshold=PRUNE_THRESHOLD, processes=PROCESSES)
    pi = np.array([[dist[a] if a in dist.domain() else 0. for a in env.agent_actions[agent.name]]
                   for dist in expert_pi])
    expert_q = get_action_values(agent, states, env.agent_actions[agent.name], horizon=HORIZON, processes=PROCESSES)
    expert_v = np.max(expert_q, axis=1)
    env.plot_policy(pi, expert_v, os.path.join(OUTPUT_DIR, f'expert-policy.{IMG_FORMAT}'), 'Expert Policy')

    # gets rewards
    logging.info('Computing expert rewards...')
    expert_r = np.array([agent.reward(state) for state in states])
    env.plot_func(expert_r, os.path.join(OUTPUT_DIR, f'expert-reward.{IMG_FORMAT}'), 'Expert Rewards')

    # generate trajectories using expert's reward and rationality
    logging.info('=================================')
    logging.info('Generating expert trajectories...')
    trajectories = env.generate_trajectories(NUM_TRAJECTORIES, TRAJ_LENGTH, agent,
                                             select=True, horizon=HORIZON, selection=EXPERT_ACT_SELECTION,
                                             threshold=PRUNE_THRESHOLD, processes=PROCESSES, seed=EXPERT_SEED)
    env.plot_trajectories(trajectories, agent, os.path.join(OUTPUT_DIR, f'expert-trajectories.{IMG_FORMAT}'),
                          'Expert Trajectories')

    # create learning algorithm and optimize reward weights
    logging.info('=================================')
    logging.info('Starting MaxEnt IRL optimization...')
    feat_matrix = env.get_location_feature_matrix(OUTER_COLORS, INNER_COLORS)
    rwd_vector = ObjectsRewardVector(env, agent, feat_matrix, OUTER_COLORS, INNER_COLORS)
    alg = MaxEntRewardLearning(
        'max-ent', agent.name, rwd_vector,
        processes=PROCESSES,
        normalize_weights=NORM_THETA,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        diff_threshold=THRESHOLD,
        decrease_rate=DECREASE_RATE,
        prune_threshold=PRUNE_THRESHOLD,
        exact=EXACT,
        num_mc_trajectories=NUM_MC_TRAJECTORIES,
        horizon=HORIZON,
        seed=LEARNING_SEED)
    # trajectories = [[(w.state, a) for w, a in t] for t in trajectories]
    result = alg.learn(trajectories, verbose=True)

    # saves results/stats
    alg.save_results(result, OUTPUT_DIR, IMG_FORMAT)

    # set learner's reward into expert for evaluation (compare to true model)
    rwd_vector.set_rewards(agent, result.stats[THETA_STR])

    # gets policy and value
    logging.info('=================================')
    logging.info('Computing learner policy & value function...')
    learner_pi = get_policy(agent, states, selection='distribution', threshold=PRUNE_THRESHOLD, processes=PROCESSES)
    pi = np.array([[dist[a] if a in dist.domain() else 0. for a in env.agent_actions[agent.name]]
                   for dist in learner_pi])
    learner_q = get_action_values(agent, states, env.agent_actions[agent.name], horizon=HORIZON, processes=PROCESSES)
    learner_v = np.max(learner_q, axis=1)
    env.plot_policy(pi, learner_v, os.path.join(OUTPUT_DIR, f'learner-policy.{IMG_FORMAT}'), 'Learner Policy')

    # gets rewards
    logging.info('Computing learner rewards...')
    learner_r = np.array([agent.reward(state) for state in states])
    env.plot_func(learner_r, os.path.join(OUTPUT_DIR, f'learner-reward.{IMG_FORMAT}'), 'Learner Rewards')

    logging.info('=================================')
    logging.info('Computing evaluation metrics...')
    logging.info(f'Policy mismatch: {policy_mismatch_prob(expert_pi, learner_pi):.3f}')
    logging.info(f'Policy divergence: {policy_divergence(expert_pi, learner_pi):.3f}')

    logging.info('\nFinished!')
