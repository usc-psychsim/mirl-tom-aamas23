import copy
import itertools
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Literal, Optional, Any, NamedTuple
from matplotlib.colors import ListedColormap
from matplotlib.markers import CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE
from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import makeTree, incrementMatrix, noChangeMatrix, thresholdRow, stateKey, VectorDistributionSet, \
    KeyedPlane, KeyedVector, rewardKey, setToConstantMatrix, equalRow, makeFuture
from model_learning.util.plot import distinct_colors
from model_learning.trajectory import generate_trajectories, log_trajectories
from model_learning import Trajectory

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class Location(NamedTuple):
    x: int
    y: int


X_FEATURE = 'x'
Y_FEATURE = 'y'
VISIT_FEATURE = 'v'

ACTION_NO_OP = 0
ACTION_RIGHT_IDX = 1
ACTION_LEFT_IDX = 2
ACTION_UP_IDX = 3
ACTION_DOWN_IDX = 4

# stores shift values for placement and markers for each action
MARKERS_INC = {
    ACTION_RIGHT_IDX: (.7, .5, CARETRIGHTBASE),
    ACTION_UP_IDX: (.5, .7, CARETUPBASE),
    ACTION_LEFT_IDX: (.3, .5, CARETLEFTBASE),
    ACTION_DOWN_IDX: (.5, .3, CARETDOWNBASE),
    ACTION_NO_OP: (.5, .5, '.')  # stand still action
}

ACTION_NAMES = {
    ACTION_RIGHT_IDX: 'right',
    ACTION_UP_IDX: 'up',
    ACTION_LEFT_IDX: 'left',
    ACTION_DOWN_IDX: 'down',
    ACTION_NO_OP: 'no-op',
}

TITLE_FONT_SIZE = 12
VALUE_CMAP = 'gray'  # 'viridis' # 'inferno'
TRAJECTORY_LINE_WIDTH = 1
LOC_FONT_SIZE = 6
LOC_FONT_COLOR = 'darkgrey'
NOTES_FONT_SIZE = 8
NOTES_FONT_COLOR = 'dimgrey'
POLICY_MARKER_COLOR = 'dimgrey'


class GridWorld(object):
    """
    Represents a simple gridworld environment in which agents can move in the 4 cardinal directions or stay in the same
    location.
    """

    def __init__(self, world: World, width: int, height: int, name: str = ''):
        """
        Creates a new gridworld.
        :param World world: the PsychSim world associated with this gridworld.
        :param int width: the number of horizontal cells.
        :param int height: the number of vertical cells.
        :param str name: the name of this gridworld, used as a suffix for features, actions, etc.
        """
        self.world = world
        self.width = width
        self.height = height
        self.name = name

        self.agent_actions: Dict[str, List[ActionSet]] = {}

    def add_agent_dynamics(self, agent: Agent) -> List[ActionSet]:
        """
        Adds the PsychSim action dynamics for the given agent to move in this gridworld.
        The 4 cardinal movement actions plus a stay-still/no-op action is added.
        Also registers those actions for later usage.
        :param Agent agent: the agent to which add the action movement dynamics.
        :rtype: list[ActionSet]
        :return: a list containing the agent's newly created actions.
        """
        assert agent.name not in self.agent_actions, f'An agent was already registered with the name \'{agent.name}\''
        self.agent_actions[agent.name] = []

        # creates agent's location feature
        x = self.world.defineState(
            agent.name, X_FEATURE + self.name, int, 0, self.width - 1, description=f'{agent.name}\'s horizontal location')
        self.world.setFeature(x, 0)
        y = self.world.defineState(
            agent.name, Y_FEATURE + self.name, int, 0, self.height - 1, description=f'{agent.name}\'s vertical location')
        self.world.setFeature(y, 0)

        # visits: Dict[int, str] = {}
        # for x_i, y_i in itertools.product(range(self.width), range(self.height)):
        #     loc_i = self.xy_to_idx(x_i, y_i)
        #     visits[loc_i] = self.world.defineState(agent.name, VISIT_FEATURE + f'{loc_i}' + self.name,
        #                                            int, 0, 2, f'{agent.name}\'s # of visits for each location')
        #     if loc_i == 0:
        #         self.world.setFeature(visits[loc_i], 1)
        #     else:
        #         self.world.setFeature(visits[loc_i], 0)

        # creates dynamics for the agent's movement (cardinal directions + no-op) with legality
        action = agent.addAction({'verb': 'move', 'action': 'wait'})
        tree = makeTree(noChangeMatrix(x))
        self.world.setDynamics(x, action, tree)
        self.agent_actions[agent.name].append(action)

        # move right
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'right'})
        legal_dict = {'if': equalRow(x, self.width - 1), True: False, False: True}
        agent.setLegal(action, makeTree(legal_dict))
        move_tree = makeTree(incrementMatrix(x, 1))
        self.world.setDynamics(x, action, move_tree)
        self.agent_actions[agent.name].append(action)

        # for loc_i, visit in enumerate(visits):
        #     visit_dict = {'if': KeyedPlane(KeyedVector({makeFuture(x): 1, y: self.width}), loc_i, 0),
        #                   True: incrementMatrix(visits[loc_i], 1),
        #                   False: noChangeMatrix(visits[loc_i])}
        #     self.world.setDynamics(visits[loc_i], action, makeTree(visit_dict))

        # move left
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'left'})
        legal_dict = {'if': equalRow(x, 0), True: False, False: True}
        agent.setLegal(action, makeTree(legal_dict))
        move_tree = makeTree(incrementMatrix(x, -1))
        self.world.setDynamics(x, action, move_tree)
        self.agent_actions[agent.name].append(action)

        # for loc_i, visit in enumerate(visits):
        #     visit_dict = {'if': KeyedPlane(KeyedVector({makeFuture(x): 1, y: self.width}), loc_i, 0),
        #                   True: incrementMatrix(visits[loc_i], 1),
        #                   False: noChangeMatrix(visits[loc_i])}
        #     self.world.setDynamics(visits[loc_i], action, makeTree(visit_dict))

        # move up
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'up'})
        legal_dict = {'if': equalRow(y, self.height - 1), True: False, False: True}
        agent.setLegal(action, makeTree(legal_dict))
        move_tree = makeTree(incrementMatrix(y, 1))
        self.world.setDynamics(y, action, move_tree)
        self.agent_actions[agent.name].append(action)

        # for loc_i, visit in enumerate(visits):
        #     visit_dict = {'if': KeyedPlane(KeyedVector({x: 1, makeFuture(y): self.width}), loc_i, 0),
        #                   True: incrementMatrix(visits[loc_i], 1),
        #                   False: noChangeMatrix(visits[loc_i])}
        #     self.world.setDynamics(visits[loc_i], action, makeTree(visit_dict))

        # move down
        action = agent.addAction({'verb': 'move' + self.name, 'action': 'down'})
        legal_dict = {'if': equalRow(y, 0), True: False, False: True}
        agent.setLegal(action, makeTree(legal_dict))
        move_tree = makeTree(incrementMatrix(y, -1))
        self.world.setDynamics(y, action, move_tree)
        self.agent_actions[agent.name].append(action)

        # for loc_i, visit in enumerate(visits):
        #     visit_dict = {'if': KeyedPlane(KeyedVector({x: 1, makeFuture(y): self.width}), loc_i, 0),
        #                   True: incrementMatrix(visits[loc_i], 1),
        #                   False: noChangeMatrix(visits[loc_i])}
        #     self.world.setDynamics(visits[loc_i], action, makeTree(visit_dict))

        return self.agent_actions[agent.name]

    def get_location_features(self, agent: Agent) -> Tuple[str, str]:
        """
        Gets the agent's (X,Y) features in the gridworld.
        :param Agent agent: the agent for which to get the location features.
        :rtype: (str,str)
        :return: a tuple containing the (X, Y) agent features.
        """
        x = stateKey(agent.name, X_FEATURE + self.name)
        y = stateKey(agent.name, Y_FEATURE + self.name)
        return x, y

    def get_visit_feature(self, agent: Agent) -> Dict[int, str]:
        """
        Gets the agent's visit feature in the gridworld.
        :param Agent agent: the agent for which to get the visit feature.
        :rtype: (int, str)
        :return: a dict containing the agent visit feature for each location.
        """
        visits: Dict[int, str] = {}
        for loc_i in range(self.width * self.height):
            visits[loc_i] = stateKey(agent.name, VISIT_FEATURE + f'{loc_i}' + self.name)
        return visits

    def get_location_plane(self,
                           agent: Agent,
                           locs: List[Location],
                           comp: Literal[KeyedPlane.COMPARISON_MAP] = KeyedPlane.COMPARISON_MAP[0]) -> KeyedPlane:
        """
        Gets a PsychSim plane for the given agent that can be used to compare it's current location against the given
        set of locations. Comparisons are made at the index level, i.e., in the left-right, bottom-up order.
        Also, comparison uses logical OR, i.e., it verifies against *any* of the given locations.
        :param Agent agent: the agent for which to get the comparison plane.
        :param list[Location] locs: a list of target XY coordinate tuples.
        :param str comp: the comparison to be made ('==', '>', or '<').
        :rtype: KeyedPlane
        :return: the plane corresponding to comparing the agent's location against the given coordinates.
        """
        assert comp in KeyedPlane.COMPARISON_MAP, \
            f'Invalid comparison provided: {comp}; valid: {KeyedPlane.COMPARISON_MAP}'

        x_feat, y_feat = self.get_location_features(agent)
        loc_idxs = {self.xy_to_idx(*loc) for loc in locs}

        # creates plane that checks if x+(y*width) equals any idx in the set
        return KeyedPlane(KeyedVector({x_feat: 1., y_feat: self.width}),
                          loc_idxs,
                          KeyedPlane.COMPARISON_MAP.index(comp))

    def idx_to_xy(self, i: int) -> Location:
        """
        Converts the given location index to XY coordinates. Indexes are taken from the left-right, bottom-up order.
        :param int i: the index of the location.
        :rtype: (int, int)
        :return: a tuple containing the XY coordinates corresponding to the given location index.
        """
        return Location(i % self.width, i // self.width)

    def xy_to_idx(self, x: int, y: int) -> int:
        """
        Converts the given XY coordinates to a location index. Indexes are taken from the left-right, bottom-up order.
        :param int x: the location's X coordinate.
        :param int y: the location's Y coordinate.
        :rtype: int
        :return: an integer corresponding to the given coordinates' location index.
        """
        return x + y * self.width

    def generate_trajectories(self,
                              n_trajectories: int,
                              trajectory_length: int,
                              agent: Agent,
                              init_feats: Optional[Dict[str, Any]] = None,
                              model: Optional[str] = None,
                              select: bool = False,
                              horizon: Optional[int] = None,
                              selection: Optional[Literal['distribution', 'random', 'uniform', 'consistent']] = None,
                              threshold: Optional[float] = None,
                              processes: Optional[int] = -1,
                              seed: int = 0,
                              verbose: bool = False,
                              use_tqdm: bool = True) -> List[Trajectory]:
        """
        Generates a number of fixed-length agent trajectories/traces/paths (state-action pairs).
        :param int n_trajectories: the number of trajectories to be generated.
        :param int trajectory_length: the length of the generated trajectories.
        :param Agent agent: the agent for which to record the actions.
        :param dict[str, Any] init_feats: the initial feature states from which to randomly initialize the
        trajectories. Each key is the name of the feature and the corresponding value is either a list with possible
        values to choose from, a single value, or `None`, in which case a random value will be picked based on the
        feature's domain.
        :param str model: the agent model used to generate the trajectories.
        :param bool select: whether to select from stochastic states after each world step.
        :param int horizon: the agent's planning horizon.
        :param str selection: the action selection criterion, to untie equal-valued actions.
        :param int processes: number of processes to use. Follows `joblib` convention.
        :param float threshold: outcomes with a likelihood below this threshold are pruned. `None` means no pruning.
        :param int seed: the seed used to initialize the random number generator.
        :param bool verbose: whether to show information at each timestep during trajectory generation.
        :param bool use_tqdm: whether to use tqdm to show progress bar during trajectory generation.
        :rtype: list[Trajectory]
        :return: the generated agent trajectories.
        """
        # get relevant features for this world (x-y location)
        x, y = self.get_location_features(agent)

        # if not specified, set random values for x, y pos
        if init_feats is None:
            init_feats = {}
        if x not in init_feats:
            init_feats[x] = None
        if y not in init_feats:
            init_feats[y] = None

        # generate trajectories starting from random locations in the gridworld
        return generate_trajectories(agent, n_trajectories, trajectory_length,
                                     init_feats, model, select, horizon, selection, threshold,
                                     processes, seed, verbose, use_tqdm)

    def set_achieve_locations_reward(self,
                                     agent: Agent,
                                     locs: List[Location],
                                     weight: float,
                                     model: Optional[str] = None):
        """
        Sets a reward to the agent such that if its current location is equal to one of the given locations it will
        receive the given value. Comparisons are made at the index level, i.e., in the left-right, bottom-up order.
        :param Agent agent: the agent for which to get set the reward.
        :param list[Location] locs: a list of target XY coordinate tuples.
        :param float weight: the weight/value associated with this reward.
        :param str model: the agent's model on which to set the reward.
        """
        agent.setReward(makeTree({'if': self.get_location_plane(agent, locs),
                                  True: setToConstantMatrix(rewardKey(agent.name), 1.),
                                  False: setToConstantMatrix(rewardKey(agent.name), 0.)}), weight, model)

    def get_all_states(self, agent: Agent) -> List[Optional[VectorDistributionSet]]:
        """
        Collects all PsychSim world states that the given agent can be in according to the gridworld's locations.
        Other PsychSim features *are not* changed, i.e., the agent does not perform any actions.
        :param Agent agent: the agent for which to get the states.
        :rtype: list[VectorDistributionSet]
        :return: a list of PsychSim states in the left-right, bottom-up order.
        """
        assert agent.world == self.world, 'Agent\'s world different from the environment\'s world!'

        old_state = copy.deepcopy(self.world.state)
        states = [None] * self.width * self.height

        # iterate through all agent positions and copy world state
        x, y = self.get_location_features(agent)
        for x_i, y_i in itertools.product(range(self.width), range(self.height)):
            self.world.setFeature(x, x_i)
            self.world.setFeature(y, y_i)
            idx = self.xy_to_idx(x_i, y_i)
            states[idx] = copy.deepcopy(self.world.state)

        # undo world state
        self.world.state = old_state
        return states

    def log_trajectories(self, trajectories: List[Trajectory], agent: Agent):
        """
        Prints the given trajectories to the log at the info level.
        :param list[Trajectory] trajectories: the set of trajectories to save, containing
        several sequences of state-action pairs.
        :param Agent agent: the agent whose location we want to log.
        :return:
        """
        if len(trajectories) == 0 or len(trajectories[0]) == 0:
            return

        x, y = self.get_location_features(self.world.agents[agent.name])
        assert x in self.world.variables, f'Agent \'{agent.name}\' does not have x location feature'
        assert y in self.world.variables, f'Agent \'{agent.name}\' does not have y location feature'

        log_trajectories(trajectories, [x, y])

    def plot(self, file_name: str, title: str = 'Environment', show: bool = False):
        """
        Generates and saves a grid plot of the environment, including the number of each state.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param bool show: whether to show the plot to the screen.
        :return:
        """
        plt.figure()
        self._plot(None, title, None)
        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info(f'Saved environment \'{title}\' plot to:\n\t{file_name}')
        if show:
            plt.show()
        plt.close()

    def plot_func(self,
                  value_func: np.ndarray,
                  file_name: str,
                  title: str = 'Environment',
                  cmap: str = VALUE_CMAP,
                  show: bool = False):
        """
        Generates ands saves a plot of the environment, including a heatmap according to the given value function.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param np.ndarray value_func: the value for each state of the environment of shape (n_states, 1).
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param str cmap: the colormap used to plot the reward function.
        :param bool show: whether to show the plot to the screen.
        """
        plt.figure()

        self._plot(value_func, title, cmap)

        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info('Saved environment \'{}\' plot to:\n\t{}'.format(title, file_name))
        if show:
            plt.show()
        plt.close()

    def plot_policy(self,
                    policy: np.ndarray,
                    value_func: np.ndarray,
                    file_name: str,
                    title: str = 'Policy',
                    cmap: str = VALUE_CMAP,
                    show: bool = False):
        """
        Generates ands saves a plot of the given policy in the environment.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param np.ndarray policy: the policy to be plotted of shape (n_states, n_actions).
        :param np.ndarray value_func: the value for each state of the environment of shape (n_states, 1).
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param str cmap: the colormap used to plot the reward function.
        :param bool show: whether to show the plot to the screen.
        """
        plt.figure()

        # first plot environment
        self._plot(value_func, title, cmap)

        # then plot max actions for each state
        for x, y in itertools.product(range(self.width), range(self.height)):
            idx = self.xy_to_idx(x, y)

            # plot marker (arrow) for each action, control alpha according to probability
            for a in range(policy.shape[1]):
                x_inc, y_inc, marker = MARKERS_INC[a]
                plt.plot(x + x_inc, y + y_inc, marker=marker, c=POLICY_MARKER_COLOR, mew=0.5, mec='0',
                         alpha=policy[idx, a])

        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info(f'Saved policy \'{title}\' plot to:\n\t{file_name}')
        if show:
            plt.show()
        plt.close()

    def plot_trajectories(self,
                          trajectories: List[Trajectory],
                          agent: Agent,
                          file_name: str,
                          title: str = 'Trajectories',
                          value_func: Optional[np.ndarray] = None,
                          cmap: str = VALUE_CMAP,
                          show: bool = False):
        """
        Plots the given set of trajectories over a representation of the environment.
        Utility method for 2D / gridworld environments that can have a visual representation.
        :param list[list[tuple[World, Distribution]]] trajectories: the set of trajectories to save, containing
        several sequences of state-action pairs.
        :param Agent agent: the agent whose location we want to log.
        :param str file_name: the path to the file in which to save the plot.
        :param str title: the title of the plot.
        :param np.ndarray value_func: the value for each state of the environment of shape (n_states, 1).
        :param str cmap: the colormap used to plot the reward function.
        :param bool show: whether to show the plot to the screen.
        """
        if len(trajectories) == 0 or len(trajectories[0]) == 0:
            return

        x, y = self.get_location_features(self.world.agents[agent.name])
        assert x in self.world.variables, f'Agent \'{agent.name}\' does not have x location feature'
        assert y in self.world.variables, f'Agent \'{agent.name}\' does not have y location feature'

        plt.figure()
        ax = plt.gca()

        # plot base environment
        self._plot(value_func, title, cmap)

        # plots trajectories
        t_colors = distinct_colors(len(trajectories))
        for i, trajectory in enumerate(trajectories):
            xs = []
            ys = []
            for t, sa in enumerate(trajectory):
                x_t = sa.world.getValue(x)
                y_t = sa.world.getValue(y)
                xs.append(x_t + .5)
                ys.append(y_t + .5)

                # plots label and final mark
                if t == len(trajectory) - 1:
                    plt.plot(x_t + .5, y_t + .5, 'x', c=t_colors[i], mew=1)
                    ax.annotate(
                        'T{:02d}'.format(i), xy=(x_t + .7, y_t + .7), fontsize=NOTES_FONT_SIZE, c=NOTES_FONT_COLOR)

            plt.plot(xs, ys, c=t_colors[i], linewidth=TRAJECTORY_LINE_WIDTH)

        plt.savefig(file_name, pad_inches=0, bbox_inches='tight')
        logging.info('Saved trajectories \'{}\' plot to:\n\t{}'.format(title, file_name))
        if show:
            plt.show()
        plt.close()

    def _plot(self, val_func: Optional[np.ndarray] = None, title: str = 'Environment', cmap: Optional[str] = None):
        ax = plt.gca()

        if val_func is None:
            # plots grid with cell numbers
            grid = np.zeros((self.height, self.width))
            plt.pcolor(grid, cmap=ListedColormap(['white']), edgecolors='darkgrey')
            for x, y in itertools.product(range(self.width), range(self.height)):
                ax.annotate('({},{})'.format(x, y), xy=(x + .05, y + .05), fontsize=LOC_FONT_SIZE, c=LOC_FONT_COLOR)
        else:
            # plots given value function as heatmap
            val_func = val_func.reshape((self.width, self.height))
            plt.pcolor(val_func, cmap=cmap)
            plt.colorbar()

        # turn off tick labels
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect('equal', adjustable='box')
        plt.title(title, fontsize=TITLE_FONT_SIZE)
