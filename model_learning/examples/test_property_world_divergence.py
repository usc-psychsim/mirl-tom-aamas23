import os
import numpy as np
import copy
from typing import List, Dict
from model_learning.environments.property_gridworld import PropertyGridWorld
from model_learning.util.io import create_clear_dir
from model_learning import StateActionPair
from psychsim.world import World
from model_learning.features import expected_feature_counts, estimate_feature_counts, LinearRewardVector
from psychsim.pwl import Distribution
from model_learning.trajectory import copy_world
from model_learning.evaluation.metrics import policy_divergence

__author__ = 'Pedro Sequeira and Haochen Wu'
__email__ = 'pedrodbs@gmail.com and hcaawu@gmail.com'

GOAL_FEATURE = 'g'
NAVI_FEATURE = 'f'

X_FEATURE = 'x'
Y_FEATURE = 'y'
PROPERTY_FEATURE = 'p'
PROPERTY_LIST = ['unknown', 'found', 'ready', 'clear', 'empty']
WORLD_NAME = 'PGWorld'

ENV_SIZE = 3
ENV_SEED = 47
NUM_EXIST = 3

TEAM_AGENTS = ['Medic', 'Explorer']
AGENT_ROLES = [{'Goal': 1}, {'Navigator': 0.5}]

HORIZON = 2  # 0 for random actions
PRUNE_THRESHOLD = 1e-2
ACT_SELECTION = 'random'
RATIONALITY = 1 / 0.1

# common params

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output/examples/test-property-world')
NUM_TRAJECTORIES = 16  # 10
TRAJ_LENGTH = 25  # 15
PROCESSES = -1
DEBUG = 0
np.set_printoptions(precision=4)
EVALUATE_BY = 'EPISODES'
# EVALUATE_BY = 'FEATURES'
# EVALUATE_BY = 'EMPIRICAL'
print(EVALUATE_BY)

if __name__ == '__main__':
    test_ag_i = [0, 1]
    print(test_ag_i)
    create_clear_dir(OUTPUT_DIR, clear=False)
    world = World()
    world.setParallel()
    env = PropertyGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_EXIST, WORLD_NAME, seed=ENV_SEED)
    print('Initializing World', f'h:{HORIZON}', f'x:{env.width}', f'y:{env.height}', f'v:{env.num_exist}')
    print('Env Seed', ENV_SEED, test_ag_i)


    # team of two agents
    team = []
    for ag_i in range(len(TEAM_AGENTS)):
        agent = world.addAgent(TEAM_AGENTS[ag_i])
        # define agent dynamics
        env.add_location_property_dynamics(agent, idle=True)
        team.append(agent)
    # collaboration dynamics
    env.add_collaboration_dynamics([agent for agent in team])

    # set agent rewards, and attributes
    team_rwd = []
    for ag_i, agent in enumerate(team):
        rwd_features, rwd_f_weights = env.get_role_reward_vector(agent, AGENT_ROLES[ag_i])
        agent_lrv = LinearRewardVector(rwd_features)
        if ag_i == 0:
            # rwd_f_weights = np.array([0, 0, 0, 0, 0, 0])  # random reward
            rwd_f_weights = np.array([0.062, 0.062, 0.187, 0.625, 0.031, 0.031])  # gt
            # rwd_f_weights = - np.array([0.062, 0.062, 0.187, 0.625, 0.031, 0.031])  # opposite
            # rwd_f_weights = np.array([0.015318192417385733, 0.1844248078482227, 0.23077641060081897,
            #                     0.24607229883978093, 0.26820367798138783, 0.055204612312403784])  # learned with gt
            # rwd_f_weights = np.array([0.01878362580308851, 0.16478999901405544, 0.2713735395498321,
            #                     0.263960212489292, 0.2260549673882416, 0.055037655755490154])  # learned with gtoprand inference
            # rwd_f_weights = np.array([0.06197305676980969, 0.07532694434692906, 0.3522639561849776,
            #                     0.44931645182316027, 0.06058269567728371, -0.1005368951978396395])  # learned with opposite
            # rwd_f_weights = np.array([0.0672, 0.0781, 0.3482, 0.4426, 0.059, -0.0048]) # learned with opposite 2
            # rwd_f_weights = np.array([0.0004, 0.1614, 0.2898, 0.2835, 0.256, 0.0089]) # learned with rdtksc inference
            # rwd_f_weights = np.array([0.10066148181050796, 0.1645613822023652, 0.2559778593118164,
            #                     0.29270815510181414, 0.1415178327488864, 0.04457328882460985])  # learned with gt inference gtoprand
        else:
            # rwd_f_weights = np.array([0, 0, 0])  # random reward
            rwd_f_weights = np.array([0.25, 0.25, 0.5])  # gt
            # rwd_f_weights = - np.array([0.25, 0.25, 0.5])  # opposite
            # rwd_f_weights = np.array([0.006554615703699425, 0.23050842848430586, 0.7629369558119946])  # learned with gt
            # rwd_f_weights = np.array([0.6151432079257714, 0.21113397863539382, 0.1737228134388347])  # learned with gtoprand inference
            # rwd_f_weights = np.array([0.6677601587685111, 0.000382152136435009, 0.331857689095054])  # learned with opposite
            # rwd_f_weights = np.array([0.6529, -0.0226, 0.3245]) # learned with opposite 2
            # rwd_f_weights = np.array([0.2627722797013279, 0.43297962649286076, 0.30424809380581147])  # learned with rdtksc inference
            # rwd_f_weights = np.array([0.5358654865179844, 0.32261899383623155, 0.14151551964578404])  # learned with gt inference gtoprand

        rwd_f_weights = np.array(rwd_f_weights) / np.linalg.norm(rwd_f_weights, 1)
        agent_lrv.set_rewards(agent, rwd_f_weights)
        print(f'{agent.name} Reward Features')
        print(agent_lrv.names, rwd_f_weights)
        team_rwd.append(agent_lrv)
        # agent.printReward(agent.get_true_model())

        agent.setAttribute('selection', ACT_SELECTION)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('rationality', RATIONALITY)
        agent.setAttribute('discount', 0.7)

    # example run
    my_turn_order = [{agent.name for agent in team}]
    env.world.setOrder(my_turn_order)

    learner_team = [copy.deepcopy(agent) for agent in team]
    learner_suffix = '_learner'

    learner_team_rwd = []
    for ag_i, agent in enumerate(learner_team):
        rwd_features, rwd_f_weights = env.get_role_reward_vector(agent, AGENT_ROLES[ag_i])
        agent_lrv = LinearRewardVector(rwd_features)
        if ag_i == 0:
            # weights = np.array([0, 0, 0, 0, 0, 0])  # random reward
            # weights = np.array([0.062, 0.062, 0.187, 0.625, 0.031, 0.031])  # gt
            # weights = - np.array([0.062, 0.062, 0.187, 0.625, 0.031, 0.031])  # opposite
            # weights = np.array([0.015318192417385733, 0.1844248078482227, 0.23077641060081897,
            #                     0.24607229883978093, 0.26820367798138783, 0.055204612312403784])  # learned with gt
            # weights = np.array([0.01878362580308851, 0.16478999901405544, 0.2713735395498321,
            #                     0.263960212489292, 0.2260549673882416, 0.055037655755490154])  # learned with gtoprand inference
            # weights = np.array([0.06197305676980969, 0.07532694434692906, 0.3522639561849776,
            #                     0.44931645182316027, 0.06058269567728371, -0.0005368951978396395])  # learned with opposite
            weights = np.array([0.0672, 0.0781, 0.3482, 0.4426, 0.059, -0.0048])  # learned with opposite 2
            # weights = np.array([0.0004, 0.1614, 0.2898, 0.2835, 0.256, 0.0089]) # learned with rdtksc inference
            # weights = np.array([0.10066148181050796, 0.1645613822023652, 0.2559778593118164,
            #                     0.29270815510181414, 0.1415178327488864, 0.04457328882460985])  # learned with gt inference gtoprand
        else:
            # weights = np.array([0, 0, 0])  # random reward
            # weights = np.array([0.25, 0.25, 0.5])  # gt
            # weights = - np.array([0.25, 0.25, 0.5])  # opposite
            # weights = np.array([0.006554615703699425, 0.23050842848430586, 0.7629369558119946])  # learned with gt
            # weights = np.array([0.6151432079257714, 0.21113397863539382, 0.1737228134388347])  # learned with gtoprand inference
            # weights = np.array([0.6677601587685111, 0.000382152136435009, 0.331857689095054])  # learned with opposite
            weights = np.array([0.6529, -0.0226, 0.3245])  # learned with opposite 2
            # weights = np.array([0.2627722797013279, 0.43297962649286076, 0.30424809380581147])  # learned with rdtksc inference
            # weights = np.array([0.5358654865179844, 0.32261899383623155, 0.14151551964578404])  # learned with gt inference gtoprand

        rwd_f_weights = np.array(weights) / np.linalg.norm(weights, 1)
        agent_lrv.set_rewards(agent, rwd_f_weights)
        # agent_lrv.rwd_weights = weights
        print(f'{agent.name + learner_suffix} Reward Features')
        print(agent_lrv.names, rwd_f_weights)
        learner_team_rwd.append(agent_lrv)

    if EVALUATE_BY == 'EMPIRICAL':
        NUM_TRAJECTORIES = 32
        team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH,
                                                           n_trajectories=NUM_TRAJECTORIES,
                                                           horizon=HORIZON, selection=ACT_SELECTION,
                                                           processes=PROCESSES,
                                                           threshold=1e-2, seed=ENV_SEED)
        team_trajs = [] * len(team)
        for ag_i, agent in enumerate(team):
            if ag_i in test_ag_i:
                agent_trajs = []
                for team_traj in team_trajectories:
                    agent_traj = []
                    for team_step in team_traj:
                        tsa = team_step
                        sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
                        agent_traj.append(sa)
                    agent_trajs.append(agent_traj)
                team_trajs.append(agent_trajs)

                feature_func = lambda s: team_rwd[ag_i].get_values(s)
                empirical_fc = expected_feature_counts(agent_trajs, feature_func)
                print(empirical_fc)

    if EVALUATE_BY == 'FEATURES':
        team_trajectories = env.generate_team_trajectories(team, TRAJ_LENGTH,
                                                           n_trajectories=NUM_TRAJECTORIES,
                                                           horizon=HORIZON, selection=ACT_SELECTION,
                                                           processes=PROCESSES,
                                                           threshold=1e-2, seed=ENV_SEED)
        team_trajs = [] * len(team)
        for ag_i, agent in enumerate(team):
            if ag_i in test_ag_i:
                agent_trajs = []
                for team_traj in team_trajectories:
                    agent_traj = []
                    for team_step in team_traj:
                        tsa = team_step
                        sa = StateActionPair(tsa.world, tsa.action[agent.name], tsa.prob)
                        agent_traj.append(sa)
                    agent_trajs.append(agent_traj)
                team_trajs.append(agent_trajs)

                feature_func = lambda s: team_rwd[ag_i].get_values(s)
                empirical_fc = expected_feature_counts(agent_trajs, feature_func)
                print('Empirical:', empirical_fc)

                initial_states = [t[0].world.state for t in agent_trajs]  # initial states for fc estimation
                learner_world = copy_world(learner_team[ag_i].world)
                learner_agent = learner_world.agents[learner_team[ag_i].name]
                learner_feature_func = lambda s: learner_team_rwd[ag_i].get_values(s)
                traj_len = len(agent_trajs[0])

                expected_fc = estimate_feature_counts(learner_agent, initial_states, traj_len, learner_feature_func,
                                                      exact=False, num_mc_trajectories=16,
                                                      horizon=HORIZON, threshold=PRUNE_THRESHOLD,
                                                      processes=PROCESSES, seed=ENV_SEED,
                                                      verbose=False, use_tqdm=True)
                print('Estimated:', expected_fc)
                diff = empirical_fc - expected_fc
                print(agent.name)
                print(f'Feature count different:', diff, f'={np.sum(np.abs(diff))}')

    if EVALUATE_BY == 'EPISODES':
        NUM_TRAJECTORIES = 32
        team_trajectories = env.generate_expert_learner_trajectories(team, learner_team, TRAJ_LENGTH,
                                                                     n_trajectories=NUM_TRAJECTORIES,
                                                                     horizon=HORIZON, selection=ACT_SELECTION,
                                                                     processes=PROCESSES,
                                                                     threshold=1e-2, seed=ENV_SEED)
        # print(team_trajectories)
        team_pi: Dict[str, List[Distribution]] = {}
        expert_and_learner = [agent.name for agent in team] + [agent.name + learner_suffix for agent in learner_team]
        for agent_name in expert_and_learner:
            team_pi[agent_name] = []
        for team_traj in team_trajectories:
            for tsa in team_traj:
                for agent_name, agent_action in tsa.action.items():
                    team_pi[agent_name].append(agent_action)
                    # print(tsa.action)
        # print(team_pi)
        for ag_i, agent in enumerate(team):
            print(agent.name)
            print(f'Policy divergence:'
                  f' {policy_divergence(team_pi[agent.name], team_pi[agent.name + learner_suffix]):.3f}')
        # env.play_team_trajectories(team_trajectories, team, OUTPUT_DIR)