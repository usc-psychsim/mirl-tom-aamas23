from model_learning.environments.property_gridworld import *

GOAL_FEATURE = 'g'
NAVI_FEATURE = 'f'
VISIT_FEATURE = 'v'
MOVEMENT = ['right', 'left', 'up', 'down']

# use this or property-action pair reward
class PropertyActionComparisonLinearRewardFeature(LinearRewardFeature):
    """
    Represents a reward feature that returns `1` under a property-action pair and `0` otherwise.
    """

    def __init__(self, name: str, agent_name: str, env: PropertyGridWorld,
                 action_value: str, property_value_next: int,
                 comparison: str):
        """
        Creates a new reward feature.
        :param str name: the label for this reward feature.
        :param Agent agent:
        :param PropertyGridWorld env: the PsychSim world capable of retrieving the feature's value given a state.
        :param str action_key: the named action key associated with this feature.
        :param str or int or float action_value: the value to be compared against the feature to determine its truth (boolean) value.
        :param str property_key: the named property key associated with this feature.
        :param int property_value_next: the value to be compared against the feature to determine its truth (boolean) value.
        :param str comparison: the comparison to be performed, one of `{'==', '>', '<'}`.
        """
        super().__init__(name)
        self.env = env
        self.world = env.world
        self.action_key = actionKey(agent_name)
        self.action_value = action_value
        self.property_value_next = property_value_next
        self.locations = list(range(env.width * env.height))

        assert comparison in KeyedPlane.COMPARISON_MAP, \
            f'Invalid comparison provided: {comparison}; valid: {KeyedPlane.COMPARISON_MAP}'
        self.comparison = KeyedPlane.COMPARISON_MAP.index(comparison)
        self.comp_func = COMPARISON_OPS[comparison]

    def get_value(self, state: State) -> float:
        # TODO # not valid for property-action pair
        # collects feature value distribution and returns weighted sum
        dist = np.array(
            [[float(self.comp_func(self.world.float2value(self.action_key, kv[self.action_key]), self.action_value)), p]
             for kv, p in state.distributions[state.keyMap[self.action_key]].items()])
        return dist[:, 0].dot(dist[:, 1]) * self.normalize_factor

    def set_reward(self, agent: Agent, weight: float, model: Optional[str] = None):
        rwd_key = rewardKey(agent.name)
        x, y = self.env.get_location_features(agent)
        property_action_tree = {'if': KeyedPlane(KeyedVector({self.action_key: 1}), self.action_value, self.comparison)}
        sub_pa_tree = {'if': KeyedPlane(KeyedVector({x: 1, y: self.env.width}), self.locations, 0)}
        for loc_i, loc in enumerate(self.locations):
            p_loc = self.env.p_state[loc]
            property_tree = {'if': KeyedPlane(KeyedVector({p_loc: 1}), self.property_value_next, self.comparison),
                             True: setToConstantMatrix(rwd_key, 1.),
                             False: setToConstantMatrix(rwd_key, 0.)}
            sub_pa_tree[loc_i] = property_tree
        sub_pa_tree[None] = setToConstantMatrix(rwd_key, 0.)
        property_action_tree[True] = sub_pa_tree
        property_action_tree[False] = setToConstantMatrix(rwd_key, 0.)
        agent.setReward(makeTree(property_action_tree), weight * self.normalize_factor, model)


# moved to property_gridworld, using regular Agent class
class AgentRoles(Agent):
    """
    setup agent roles and the corresponding reward features
    """

    def __init__(self, name: str, roles: Dict[str, float] = None):
        super().__init__(name, world=None)
        if roles is None:
            roles = {'Goal': 1}
        self.roles = roles

    def get_role_reward_vector(self, env: PropertyGridWorld):
        reward_features = []
        rf_weights = []
        if 'Goal' in self.roles:  # scale -1 to 1
            d2c = env.get_d2c_feature(self)
            r_d2c = NumericLinearRewardFeature(DISTANCE2CLEAR_FEATURE, d2c)
            reward_features.append(r_d2c)
            rf_weights.append(0.2 * self.roles['Goal'])

            # if self.track_feature:
            #     r_goal = NumericLinearRewardFeature(GOAL_FEATURE, stateKey(WORLD, GOAL_FEATURE))
            #     reward_features.append(r_goal)
            #     rf_weights.append(self.roles['Goal'])

            rescue_action = self.find_action({'action': 'rescue'})
            r_rescue = ActionLinearRewardFeature('rescue', self, rescue_action)
            reward_features.append(r_rescue)
            rf_weights.append(0.3 * self.roles['Goal'])

            evacuate_action = self.find_action({'action': 'evacuate'})
            r_evacuate = ActionLinearRewardFeature('evacuate', self, evacuate_action)
            reward_features.append(r_evacuate)
            rf_weights.append(self.roles['Goal'])

            nowhere_action = self.find_action({'action': 'nowhere'})
            r_nowhere = ActionLinearRewardFeature('nowhere', self, nowhere_action)
            reward_features.append(r_nowhere)
            rf_weights.append(0.1 * self.roles['Goal'])

            call_action = self.find_action({'action': 'call'})
            r_call = ActionLinearRewardFeature('call', self, call_action)
            reward_features.append(r_call)
            rf_weights.append(0.1 * self.roles['Goal'])

            for act in {'right', 'left', 'up', 'down', 'nowhere', 'search', 'rescue', 'evacuate'}:
                action = self.find_action({'action': act})
                env.world.setDynamics(env.m, action, makeTree(setToConstantMatrix(env.m, -1)))

            # for move in MOVEMENT:
            #     move_action = self.find_action({'action': move})
            #     r_move = ActionLinearRewardFeature(move, self, move_action)
            #     reward_features.append(r_move)
            #     rf_weights.append(-0.01 * self.roles['Goal'])

        if 'Navigator' in self.roles:
            d2h = env.get_d2h_feature(self)
            r_d2h = NumericLinearRewardFeature(DISTANCE2HELP_FEATURE, d2h)
            reward_features.append(r_d2h)
            rf_weights.append(self.roles['Navigator'])

            search_action = self.find_action({'action': 'search'})
            r_search = ActionLinearRewardFeature('search', self, search_action)
            reward_features.append(r_search)
            rf_weights.append(self.roles['Navigator'])

            # if env.track_feature:
            #     f = env.get_navi_features(self)
            #     r_navi = NumericLinearRewardFeature(NAVI_FEATURE, f)
            #     reward_features.append(r_navi)
            #     rf_weights.append(self.roles['Navigator'])

            evacuate_action = self.find_action({'action': 'evacuate'})
            r_evacuate = ActionLinearRewardFeature('evacuate', self, evacuate_action)
            reward_features.append(r_evacuate)
            rf_weights.append(2 * self.roles['Navigator'])

            env.remove_action(self, 'nowhere')
            env.remove_action(self, 'call')


            # visits = env.get_visit_feature(self)
            # for loc_i in range(env.width*env.height):
            #     r_visit = ValueComparisonLinearRewardFeature(
            #         VISIT_FEATURE+f'{loc_i}', env.world, visits[loc_i], 2, '==')
            #     reward_features.append(r_visit)
            #     rf_weights.append(-.02)

        if 'SubGoal' in self.roles:  # small reward for sub-goal: rescue when found
            # rescue_action = self.find_action({'action': 'rescue'})
            # r_rescue_found = PropertyActionComparisonLinearRewardFeature(
            #     'rescue_found', self.name, env, rescue_action, 2, '==')
            # reward_features.append(r_rescue_found)
            # rf_weights.append(0.1)

            rescue_action = self.find_action({'action': 'rescue'})
            r_rescue = ActionLinearRewardFeature('rescue', self, rescue_action)
            reward_features.append(r_rescue)
            rf_weights.append(self.roles['SubGoal'])

            evacuate_action = self.find_action({'action': 'evacuate'})
            r_evacuate = ActionLinearRewardFeature('evacuate', self, evacuate_action)
            reward_features.append(r_evacuate)
            rf_weights.append(self.roles['SubGoal'])

        return reward_features, rf_weights


# moved to property_gridworld, using regular LinearRewardVector class
class AgentLinearRewardVector(LinearRewardVector):
    """
    same as LinearRewardVector now, automatically set reward
    """

    def __init__(self, agent: Agent, rf: List[LinearRewardFeature],
                 weights: np.ndarray, model: Optional[str] = None):
        super().__init__(rf)
        self.rwd_features = rf
        self.rwd_weights = weights
        # self.set_rewards(agent, weights, model)

    # def set_rewards(self, agent: Agent, weights: np.ndarray, model: Optional[str] = None):
    #
    #     assert len(weights) == len(self.rwd_features), \
    #         'Provided weight vector\'s dimension does not match reward features'
    #
    #     agent.setAttribute('R', {}, model)  # make sure to clear agent's reward function
    #     for i, weight in enumerate(weights):
    #         self.rwd_features[i].set_reward(agent, weight, model)
