import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, NamedTuple, Optional
from psychsim.world import World
from psychsim.agent import Agent
from model_learning.environments.gridworld import GridWorld, Location
from model_learning.util.plot import distinct_colors

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


class Color(NamedTuple):
    inner: int
    outer: int


class ObjectsGridWorld(GridWorld):
    """
    Represents a gridworld environment containing a set of objects. Each object has a inner and an outer color, which,
    e.g., can be used to manipulate rewards provided to an agent.
    """

    def __init__(self, world: World, width: int, height: int, num_objects: int, num_colors: int,
                 name: str = '', seed: int = 0, show_objects: bool = True, single_color: bool = True):
        """
        Creates a new gridworld.
        :param World world: the PsychSim world associated with this gridworld.
        :param int width: the number of horizontal cells.
        :param int height: the number of vertical cells.
        :param str name: the name of this gridworld, used as a suffix for features, actions, etc.
        :param int num_objects: the number of objects to place in the environment.
        :param int num_colors: the number of (inner and outer) colors available for the objects.
        :param int seed: the seed used to initialize the random number generator to create and place the objects.
        :param bool show_objects: whether to show objects when plotting the environment. Can be changed at any time.
        :param bool single_color: whether objects should have a single color (i.e., inner and outer are the same).
        """
        super().__init__(world, width, height, name)
        self.num_objects = num_objects
        self.num_colors = num_colors
        self.show_objects = show_objects

        # initialize objects
        rng = np.random.RandomState(seed)
        locations = rng.choice(np.arange(width * height), num_objects, False)
        self.objects: Dict[Location, Color] = {}
        for loc in locations:
            x, y = self.idx_to_xy(loc)
            color = rng.randint(num_colors)
            self.objects[Location(x, y)] = Color(color, color) if single_color else Color(color, rng.randint(num_colors))

        # gets locations indexed by color
        self.inner_locs: List[List[Location]] = [[] for _ in range(self.num_colors)]
        self.outer_locs: List[List[Location]] = [[] for _ in range(self.num_colors)]
        for loc, color in self.objects.items():
            self.inner_locs[color.inner].append(loc)
            self.outer_locs[color.outer].append(loc)

    def _plot(self, val_func: Optional[np.ndarray] = None, title: str = 'Environment', cmap: Optional[str] = None):

        super()._plot(val_func, title, cmap)

        if self.show_objects:
            ax = plt.gca()

            # adds colors legend
            obj_colors = distinct_colors(self.num_colors)
            patches = []
            for i, color in enumerate(obj_colors):
                patches.append(mpatches.Patch(color=color, label=f'Color {i}'))
            leg = plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(-.35, 0.02, 1, 1), fancybox=False)
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(0.8)

            # plots objects as colored circles
            for loc, color in self.objects.items():
                outer_circle = plt.Circle((loc[0] + .5, loc[1] + .5), 0.3, color=obj_colors[color[0]])
                inner_circle = plt.Circle((loc[0] + .5, loc[1] + .5), 0.1, color=obj_colors[color[1]])
                ax.add_artist(outer_circle)
                ax.add_artist(inner_circle)

    def get_location_feature_matrix(self, outer: bool = True, inner: bool = True) -> np.ndarray:
        """
        Gets a matrix containing boolean features for the presence of object colors in each location in the environment,
        where a feature is `1` if there is an object of the corresponding color in that location, `0` otherwise
        :param bool outer: whether to include features for the presence of objects' outer colors.
        :param bool inner: whether to include features for the presence of objects' inner colors.
        :rtype: np.ndarray
        :return: an array of shape (width, height, num_colors [, num_colors]) containing the color features for each
        environment location.
        """
        assert inner or outer, 'At least one option "inner" or "outer" must be true'

        if inner and outer:
            feat_matrix = np.zeros((self.width, self.height, self.num_colors, self.num_colors), dtype=np.bool)
        else:
            feat_matrix = np.zeros((self.width, self.height, self.num_colors), dtype=np.bool)

        for loc, color in self.objects.items():
            if inner and outer:
                feat_matrix[loc.x, loc.y, color.inner, color.outer] = 1
            elif outer:
                feat_matrix[loc.x, loc.y, color.outer] = 1
            else:
                feat_matrix[loc.x, loc.y, color.inner] = 1
        return feat_matrix

    def get_distance_feature_matrix(self, outer: bool = True, inner: bool = True) -> np.ndarray:
        """
        Gets a matrix containing numerical features denoting the distance to the nearest object inner or outer color
        from each location in the environment.
        :param bool outer: whether to include features for the nearest distance to objects' outer colors.
        :param bool inner: whether to include features for the nearest distance to objects' inner colors.
        :rtype: np.ndarray
        :return: an array of shape (width, height, num_colors [, num_colors]) containing the color features for each
        environment location.
        """
        assert inner or outer, 'At least one option "inner" or "outer" must be true'

        if inner and outer:
            feat_matrix = np.full((self.width, self.height, self.num_colors, self.num_colors), np.inf)
        else:
            feat_matrix = np.full((self.width, self.height, self.num_colors), np.inf)

        for loc in itertools.product(range(self.width), range(self.height)):
            x, y = loc
            for obj_loc, color in self.objects.items():
                dist = np.linalg.norm(np.array(loc) - obj_loc)
                if inner and outer:
                    feat_matrix[x, y, color.inner, color.outer] = min(feat_matrix[x, y, color.inner, color.outer], dist)
                elif outer:
                    feat_matrix[x, y, color.outer] = min(feat_matrix[x, y, color.outer], dist)
                else:
                    feat_matrix[x, y, color.inner] = min(feat_matrix[x, y, color.inner], dist)
        feat_matrix[np.isinf(feat_matrix)] = np.nan
        return feat_matrix

    def set_linear_color_reward(self,
                                agent: Agent,
                                weights_outer: Optional[np.ndarray] = None,
                                weights_inner: Optional[np.ndarray] = None,
                                model: Optional[str] = None):
        """
        Sets a reward to the agent that is a linear combination of the given weights associated with each object color.
        In other words, when the agent is collocated with some object, the received reward will be proportional to the
        value associated with that object.
        :param Agent agent: the agent for which to get set the reward.
        :param np.ndarray or list[float] weights_outer: the reward weights associated with each object's outer color.
        :param np.ndarray or list[float] weights_inner: the reward weights associated with each object's inner color.
        :param str model: the agent's model on which to set the reward.
        """
        assert weights_outer is not None or weights_inner is not None, \
            'At least one option "weights_inner" or "weights_outer" must be given'

        # checks num colors
        assert weights_outer is None or len(weights_outer) == self.num_colors, \
            f'Weight vectors should have length {self.num_colors}.'
        assert weights_inner is None or len(weights_inner) == self.num_colors, \
            f'Weight vectors should have length {self.num_colors}.'

        agent.setAttribute('R', {}, model)  # make sure to clear agent's reward function

        # sets the corresponding weight for each object location
        for c in range(self.num_colors):
            if weights_outer is not None:
                self.set_achieve_locations_reward(agent, self.outer_locs[c], weights_outer[c], model)
            if weights_inner is not None:
                self.set_achieve_locations_reward(agent, self.inner_locs[c], weights_inner[c], model)
