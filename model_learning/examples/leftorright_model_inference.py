import random
import time

from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import makeTree, modelKey, incrementMatrix, setToConstantMatrix, rewardKey, stateKey
from psychsim.reward import maximizeFeature, minimizeFeature

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of inference over reward models of another agent based on observations of their behavior.' \
                  'Two agents (actor1 and action2) moves in the environment (always right or left).' \
                  'Another agent (the observer) maintains a belief over the actors\' reward model (either an agent ' \
                  'that prefers going right, going left or acting randomly. This belief is updated based on ' \
                  'observations of the actors\' actions over time.'

# parameters
MAX_STEPS = 5
AGENT_NAMES = ['actor1', 'actor2']
OBSERVER_NAME = 'observer'
HORIZON = 2
AGENT_SELECTION = 'random'
MODEL_RATIONALITY = 0.5
SEED = 17


def _get_belief(feature: str, ag: Agent, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(ag.name, unique=True)
    return world.getFeature(feature, state=ag.getBelief(model=model))


if __name__ == '__main__':
    random.seed(SEED)

    # create world and add actor agent
    world = World()

    # for each agent
    team = []
    for ag_i, AGENT_NAME in enumerate(AGENT_NAMES):
        agent = world.addAgent(AGENT_NAME)
        agent.setAttribute('horizon', HORIZON)
        agent.setAttribute('selection', AGENT_SELECTION)

        # add location variable
        loc = world.defineState(agent.name, 'location', int, -1000, 1000, 'Agent\'s location')
        world.setFeature(loc, 0)

        # define agents' actions (left and right)
        action = agent.addAction({'verb': 'move', 'action': 'left'})
        tree = makeTree(incrementMatrix(loc, -1))
        world.setDynamics(loc, action, tree)
        action = agent.addAction({'verb': 'move', 'action': 'right'})
        tree = makeTree(incrementMatrix(loc, 1))
        world.setDynamics(loc, action, tree)

        # define true reward (1 maximize loc 2 minimize loc)
        if ag_i == 0:
            agent.setReward(maximizeFeature(loc, agent.name), 1)
        else:
            agent.setReward(minimizeFeature(loc, agent.name), 1)

        # add agent models (prefer positive vs negative location vs random agent)
        true_model = agent.get_true_model()
        prefer_pos_model = f'{agent.name}_' + 'prefer_positive_loc'
        agent.addModel(prefer_pos_model, parent=true_model)
        agent.setReward(maximizeFeature(loc, agent.name), 1., model=prefer_pos_model)

        prefer_neg_model = f'{agent.name}_' + 'prefer_negative_loc'
        agent.addModel(prefer_neg_model, parent=true_model)
        agent.setReward(minimizeFeature(loc, agent.name), 1., model=prefer_neg_model)

        prefer_nothing = f'{agent.name}_' + 'prefer_nothing'  # random agent
        agent.addModel(prefer_nothing, parent=true_model)
        agent.setReward(setToConstantMatrix(rewardKey(agent.name), 0.), 1., model=prefer_nothing)

        # null_model = agent.zero_level(sample=True)  # TODO this seems to lead to a leak into the agent's true model?
        team.append(agent)

    world.setOrder([{agent.name for agent in team}])
    # create observer (no actions)
    observer = world.addAgent(OBSERVER_NAME)
    for agent in team:
        true_model = agent.get_true_model()
        # set uniform belief over agent's model in observer
        model_names = [name for name in agent.models.keys() if name != true_model]
        dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
        world.setMentalModel(observer.name, agent.name, dist)

        # agent models ignore the observer
        agent.ignore(observer.name)
        for model in model_names:
            # make models less rational to get smoother (more cautious) inference
            agent.setAttribute('rationality', MODEL_RATIONALITY, model=model)
            agent.setAttribute('selection', 'distribution', model=model)  # also set selection to distribution
            agent.setAttribute('horizon', HORIZON, model=model)
            agent.setAttribute('beliefs', True, model=model)
            agent.ignore(observer.name, model=model)

    # observer does not observe agent's true model
    observer.set_observations()

    for agent in team:
        agent_model = modelKey(agent.name)
        loc = stateKey(agent.name, 'location')
        print('====================================')
        print(f'Initial {agent.name} loc: {world.getFeature(loc)}')
        print(f'Initial belief about {agent.name}\'s model:\n{_get_belief(agent_model, observer)}')

    start_time = time.time()
    for i in range(MAX_STEPS):
        print('====================================')
        print('Step:', i)
        step = world.step()
        for agent in team:
            agent_model = modelKey(agent.name)
            loc = stateKey(agent.name, 'location')
            print(f'Current {agent.name} loc: {world.getFeature(loc)}')
            print(f'Updated belief about {agent.name}\'s model:\n{_get_belief(agent_model, observer)}')
        print(f'Time spent: {time.time()-start_time: .2f}s')
        start_time = time.time()
