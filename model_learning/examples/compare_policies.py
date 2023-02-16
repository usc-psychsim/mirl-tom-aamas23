import random
import numpy as np
from model_learning.planning import get_policy
from model_learning.environments.objects_gridworld import ObjectsGridWorld
from model_learning.evaluation.metrics import policy_mismatch_prob, policy_divergence
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Computes IRL evaluation metrics between 2 agents parameterized with a different reward function.'

ENV_SIZE = 10
NUM_OBJECTS = 25
NUM_COLORS = 5

AGENT_NAME = 'Agent'
HORIZON = 3
RATIONALITY = 1 / 0.1  # inverse temperature
SELECTION = 'distribution'
# SELECTION = 'random'
PRUNE_THRESHOLD = 1e-2

SEED = 17
PROCESSES = -1


def get_agent_policy(rwd_weights: np.ndarray):
    print('===============================================')
    # create world and agent
    world = World()
    agent = world.addAgent(AGENT_NAME)
    agent.setAttribute('selection', SELECTION)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('rationality', RATIONALITY)

    # create grid-world and add world dynamics to agent
    env = ObjectsGridWorld(world, ENV_SIZE, ENV_SIZE, NUM_OBJECTS, NUM_COLORS, seed=SEED)
    env.add_agent_dynamics(agent)
    env.set_linear_color_reward(agent, rwd_weights)
    print(f'Set reward weights to agent: {rwd_weights}')
    print(f'Reward function:\n{agent.getReward()[agent.get_true_model()]}')

    world.setOrder([{agent.name}])
    random.seed(SEED)

    print('Computing policy...')
    return get_policy(agent, env.get_all_states(agent), threshold=PRUNE_THRESHOLD, processes=PROCESSES)


if __name__ == '__main__':
    pi1 = get_agent_policy(np.array([0.5, -0.4, 0.1, 0., 0.]))
    pi2 = get_agent_policy(np.array([0.8, -0.1, 0.1, -0.01, -0.02]))

    print('===============================================')
    print(f'Policy mismatch: {policy_mismatch_prob(pi1, pi2):.3f}')
    print(f'Policy divergence: {policy_divergence(pi1, pi2):.3f}')
