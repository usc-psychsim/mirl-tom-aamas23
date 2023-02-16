import logging
import os
import numpy as np
from psychsim.world import World
from model_learning.algorithms.max_entropy import MaxEntRewardLearning, ModelLearningResult
from model_learning.features.propertyworld import AgentRoles, AgentLinearRewardVector
from model_learning.environments.property_gridworld import PropertyGridWorld
from model_learning.util.logging import change_log_handler
from model_learning.util.io import create_clear_dir
from model_learning import StateActionPair
from typing import List

__author__ = 'Pedro Sequeira and Haochen Wu'
__email__ = 'pedrodbs@gmail.com and hcaawu@gmail.com'
__description__ = 'Performs Multiagent IRL (reward model learning) in the Property World using MaxEnt IRL.'

# env params
GOAL_FEATURE = 'g'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 48
NUM_EXIST = 3

# expert params
TEAM_AGENTS = ['AHA', 'Helper1']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]

EXPERT_RATIONALITY = 1/0.1  # inverse temperature
EXPERT_ACT_SELECTION = 'random'
EXPERT_SEED = 17
NUM_TRAJECTORIES = 16 #3
TRAJ_LENGTH = 25

# learning params
NORM_THETA = True
TEAM_LEARNING_RATE = [5e-2, 1e-1]  # 0.05
MAX_EPOCHS = 100
THRESHOLD = 5e-3
DECREASE_RATE = True
EXACT = False
NUM_MC_TRAJECTORIES = 16 #10
LEARNING_SEED = 17

# common params
HORIZON = 2
PRUNE_THRESHOLD = 1e-2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/multiagent-property-world')
PROCESSES = -1
VERBOSE = True
np.set_printoptions(precision=3)


def multi_agent_reward_learning(alg: MaxEntRewardLearning,
                                agent_trajs: List[List[StateActionPair]],
                                verbose: bool) -> ModelLearningResult:
    result = alg.learn(agent_trajs, verbose=verbose)
    return result


if __name__ == '__main__':
    learner_ag_i = 0
    print(learner_ag_i)
    # create output
    create_clear_dir(OUTPUT_DIR, clear=False)
    change_log_handler(os.path.join(OUTPUT_DIR, f'mairl_gf_learner{learner_ag_i+1}.log'), logging.INFO)

    # create world and objects environment
    world = World()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Process:', PROCESSES, 'Traj Length', TRAJ_LENGTH)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')
    print('Output:', OUTPUT_DIR)

    # team of two agents
    team = []
    for i in range(len(TEAM_AGENTS)):
        team.append(world.addAgent(AgentRoles(TEAM_AGENTS[i], AGENT_ROLES[i])))

    # # define agent dynamics
    for agent in team:
        env.add_location_property_dynamics(agent, idle=True)
    env.add_collaboration_dynamics([agent for agent in team])

    team_rwd = []
    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = agent.get_role_reward_vector(env)
        agent_lrv = AgentLinearRewardVector(agent, rwd_features, rwd_f_weights)
        agent_lrv.rwd_weights = np.array(agent_lrv.rwd_weights) / np.linalg.norm(agent_lrv.rwd_weights, 1)
        agent_lrv.set_rewards(agent, agent_lrv.rwd_weights)
        logging.info(f'{agent.name} Reward Features')
        logging.info(agent_lrv.names)
        logging.info(agent_lrv.rwd_weights)
        team_rwd.append(agent_lrv)
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', EXPERT_ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', EXPERT_RATIONALITY)
        agent.setAttribute('discount', 0.7)

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    # generate trajectories using expert's reward and rationality
    logging.info('=================================')
    logging.info('Generating expert trajectories...')
    team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH, n_trajectories=NUM_TRAJECTORIES,
                                                       horizon=HORIZON, selection=EXPERT_ACT_SELECTION,
                                                       processes=PROCESSES,
                                                       threshold=1e-2, seed=ENV_SEED)
    # env.play_team_trajectories(team_trajectories, team, OUTPUT_DIR)

    # create learning algorithm and optimize reward weights
    logging.info('=================================')
    logging.info('Starting MaxEnt IRL optimization...')

    team_trajs = []
    team_algs = []
    for ag_i, agent in enumerate(team):
        if ag_i == learner_ag_i:
            rwd_vector = team_rwd[ag_i]
            agent_trajs = []
            for team_traj in team_trajectories:
                agent_traj = []
                for team_step in team_traj:
                    tsa = team_step
                    sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
                    agent_traj.append(sa)
                agent_trajs.append(agent_traj)
            team_trajs.append(agent_trajs)

            LEARNING_RATE = TEAM_LEARNING_RATE[learner_ag_i]
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
            result = alg.learn(agent_trajs, verbose=True)
        # team_algs.append(alg)

    # args = [(team_algs[t], team_trajs[t], True)
    #         for t in range(len(team))]
    # results = run_parallel(multi_agent_reward_learning, args, processes=PROCESSES, use_tqdm=True)

    # result = alg.learn(agent_trajs, verbose=True)
