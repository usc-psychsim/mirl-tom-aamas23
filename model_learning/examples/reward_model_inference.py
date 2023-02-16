import os
import logging
from model_learning.inference import track_reward_model_inference
from model_learning.util.plot import plot_evolution
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import modelKey, makeTree, setToConstantMatrix, rewardKey, actionKey
from psychsim.reward import maximizeFeature
from psychsim.helper_functions import get_true_model_name
from model_learning.environments.gridworld import GridWorld
from model_learning.util.io import create_clear_dir

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Perform reward model inference in the normal gridworld.' \
                  'There is one moving agent whose reward function is to reach the center of the grid.' \
                  'There is an observer agent that has 3 models of the moving agent (uniform prior):' \
                  '  - "middle_loc", i.e., the model with the true reward function;' \
                  '  - "maximize_loc", a model with a reward function that maximizes the coordinates values;' \
                  '  - "zero_rwd", a model with a zero reward function, resulting in a random behavior.' \
                  'The world is updated for some steps, where observer updates its belief over the models of the ' \
                  'moving agent via PsychSim inference. A plot is show with the inference evolution.'

ENV_SIZE = 10
NUM_STEPS = 100

OBSERVER_NAME = 'observer'
AGENT_NAME = 'agent'
MIDDLE_LOC_MODEL = 'middle_loc'
MAXIMIZE_LOC_MODEL = 'maximize_loc'
RANDOM_MODEL = 'zero_rwd'
MODEL_RATIONALITY = .5

HORIZON = 2
MODEL_SELECTION = 'distribution'  # TODO 'consistent' or 'random' gives an error
AGENT_SELECTION = 'random'

OUTPUT_DIR = 'output/examples/reward-model-inference'
DEBUG = False
SHOW = True
INCLUDE_RANDOM_MODEL = True


def _get_fancy_name(name):
    return name.title().replace('_', ' ')


if __name__ == '__main__':
    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create output
    create_clear_dir(OUTPUT_DIR)

    # create world, agent and observer
    world = World()
    agent = world.addAgent(AGENT_NAME)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('selection', AGENT_SELECTION)
    observer = world.addAgent(OBSERVER_NAME)
    observer.addAction('dummy')

    # create grid-world and add world dynamics to agent
    env = GridWorld(world, ENV_SIZE, ENV_SIZE)
    env.add_agent_dynamics(agent)

    # set true reward function (achieve middle location)
    x, y = env.get_location_features(agent)
    env.set_achieve_locations_reward(agent, [(4, 4)], 1.)

    world.setOrder([{agent.name, observer.name}])

    # observer does not model itself
    # observer.resetBelief(ignore={modelKey(observer.name)})
    # observer.resetBelief()


    # agent does not model itself and sees everything except true models and its reward
    # agent.resetBelief(ignore={modelKey(observer.name)})
    # agent.omega = [key for key in world.state.keys()
    #                if key not in {rewardKey(agent.name), modelKey(observer.name)}]

    # get the canonical name of the "true" agent model
    true_model = agent.get_true_model()

    # agent's models
    agent.addModel(MIDDLE_LOC_MODEL, parent=true_model, rationality=MODEL_RATIONALITY, selection=MODEL_SELECTION)
    env.set_achieve_locations_reward(agent, [(4, 4)], 1., MIDDLE_LOC_MODEL)

    agent.addModel(MAXIMIZE_LOC_MODEL, parent=true_model, rationality=MODEL_RATIONALITY, selection=MODEL_SELECTION)
    agent.setReward(maximizeFeature(x, agent.name), 1., MAXIMIZE_LOC_MODEL)
    agent.setReward(maximizeFeature(y, agent.name), 1., MAXIMIZE_LOC_MODEL)

    if INCLUDE_RANDOM_MODEL:
        # RANDOM_MODEL = agent.zero_level(sample=True)
        agent.addModel(RANDOM_MODEL, parent=true_model, rationality=MODEL_RATIONALITY, selection=MODEL_SELECTION)
        agent.setReward(makeTree(setToConstantMatrix(rewardKey(agent.name), 0)), model=RANDOM_MODEL)

    model_names = [name for name in agent.models.keys() if name != true_model]

    # observer has uniform prior distribution over possible agent models
    world.setMentalModel(observer.name, agent.name,
                         Distribution({name: 1. / (len(agent.models) - 1) for name in model_names}))

    # agent models ignore the observer
    agent.ignore(observer.name)
    for model in model_names:
        agent.setAttribute('beliefs', True, model=model)
        agent.ignore(observer.name, model=model)

    # observer does not observe agent's true model
    observer.set_observations()

    # generates trajectory
    logging.info('Generating trajectory of length {}...'.format(NUM_STEPS))
    x, y = env.get_location_features(agent)
    trajectory = env.generate_trajectories(1, NUM_STEPS, agent, {x: 0, y: 0})[0]
    print(trajectory)
    env.plot_trajectories([trajectory], agent, os.path.join(OUTPUT_DIR, 'trajectory.png'), 'Agent Path')

    # gets evolution of inference over reward models of the agent
    probs = track_reward_model_inference(trajectory, model_names, agent, observer, [x, y])

    # create and save inference evolution plot
    plot_evolution(probs.T, [_get_fancy_name(name) for name in model_names],
                   'Evolution of Model Inference', None,
                   os.path.join(OUTPUT_DIR, 'inference.png'), 'Time', 'Model Probability', True)
