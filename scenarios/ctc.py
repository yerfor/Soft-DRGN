import numba
import numpy as np
import math
from random import randint
from utils.hparams import hparams

NUM_GRIDS = hparams['env_num_grid']
NUM_HUNTERS = hparams['env_num_hunter']
NUM_RED_BANKS = hparams['env_num_red_bank']
NUM_BLUE_BANKS = hparams['env_num_blue_bank']
NUM_RED_RESOURCES = hparams['env_num_red_resource']
NUM_BLUE_RESOURCES = hparams['env_num_blue_resource']
OBS_RANGE = hparams['env_observe_range']
COMM_RANGE = hparams['env_comm_range']


@numba.jit
def is_legal(x, y):
    return (x >= OBS_RANGE) and (x <= NUM_GRIDS - OBS_RANGE - 1) and (y >= OBS_RANGE) and (
            y <= NUM_GRIDS - OBS_RANGE - 1)


@numba.jit
def decimal2binary_lst(x, length=7):
    h = []
    for _ in range(length):  # int类型的位置信息的二进制编码
        h.append(x % 2)
        x = x // 2
    return h


@numba.jit
def hunter_see_local(x_t, y_t, maze_hunters, maze_red_banks, maze_blue_banks, maze_red_resources,
                     maze_blue_resources, maze_red_treasures, maze_blue_treasures):
    h = []
    for i in range(-OBS_RANGE, OBS_RANGE + 1):  # obs range
        for j in range(-OBS_RANGE, OBS_RANGE + 1):
            h.append(maze_hunters[x_t + i][y_t + j])
            h.append(maze_red_banks[x_t + i][y_t + j])
            h.append(maze_blue_banks[x_t + i][y_t + j])
            h.append(maze_red_resources[x_t + i][y_t + j])
            h.append(maze_blue_resources[x_t + i][y_t + j])
            h.append(maze_red_treasures[x_t + i][y_t + j])
            h.append(maze_blue_treasures[x_t + i][y_t + j])
    return h


@numba.jit
def bank_see_local(x_t, y_t, maze_hunters, maze_banks, maze_resources, maze_treasures, maze_collected_treasures):
    h = []
    for i in range(-OBS_RANGE, OBS_RANGE + 1):  # obs range
        for j in range(-OBS_RANGE, OBS_RANGE + 1):
            h.append(maze_hunters[x_t + i][y_t + j])
            h.append(maze_banks[x_t + i][y_t + j])
            h.append(maze_resources[x_t + i][y_t + j])
            h.append(maze_treasures[x_t + i][y_t + j])
            h.append(maze_collected_treasures[x_t + i][y_t + j])
    return h


# @numba.jit
def cal_adj(pos_group0, pos_group1, adj):
    n_agent_group0 = pos_group0.shape[0]
    n_agent_group1 = pos_group1.shape[0]
    for i in range(n_agent_group0):
        x_i, y_i = pos_group0[i]
        for j in range(n_agent_group1):
            x_j, y_j = pos_group1[j]
            if abs(x_i - x_j) <= COMM_RANGE and abs(y_i - y_j) <= COMM_RANGE:
                adj[i, j] = 1
    return adj


class CooperativeTreasureCollection:
    def __init__(self):
        self.num_grid = NUM_GRIDS
        self.n_hunter = NUM_HUNTERS
        self.len_obs = (OBS_RANGE * OBS_RANGE) * 7 + 7  # hunter1 + bank2 + treasure2 + resource2
        self.n_red_bank = NUM_RED_BANKS
        self.n_blue_bank = NUM_BLUE_BANKS
        self.n_red_resource = NUM_RED_RESOURCES
        self.n_blue_resource = NUM_BLUE_RESOURCES
        self.n_red_treasure_capacity = self.n_hunter * 2  # how many red treasure can maintain at the meantime
        self.n_blue_treasure_capacity = self.n_hunter * 2  # how many red treasure can maintain at the meantime
        self.scatter_range = 5  # the maximum of the range that resource scatter the treasure

        self.dim_hunter_action = 7  # move5 + pick1 + dump1
        self.dim_bank_action = 5  # move5
        self.dim_hunter_obs = 359
        self.dim_bank_obs = 259
        self.obs_dim = [self.dim_hunter_obs, self.dim_bank_obs, self.dim_bank_obs]
        self.act_dim = [self.dim_hunter_action, self.dim_bank_action, self.dim_bank_action]
        self.max_keep_treasure = 5  # how many treasure a hunter can keep

        self.reward_factor_for_collect_a_treasure = 0.2
        self.reward_factor_for_dump_a_treasure = 1

        self.reset()

    def get_mazes(self):
        ret = (self.maze_hunters, self.maze_red_banks, self.maze_blue_banks, self.maze_red_resources,
               self.maze_blue_resources, self.maze_red_treasures, self.maze_blue_treasures)
        return ret

    def reset(self):
        self.maze = self.build_env()
        # Initialize Hunters
        self.pos_hunters = [
            [randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1), randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)] for _ in
            range(self.n_hunter)]
        self.maze_hunters = np.zeros([self.num_grid, self.num_grid])
        for pos_hunter in self.pos_hunters:
            x, y = pos_hunter
            self.maze_hunters[x, y] = self.maze_hunters[x, y] + 1
        self.num_red_treasure_keep_by_hunters = [0 for _ in range(self.n_hunter)]
        self.num_blue_treasure_keep_by_hunters = [0 for _ in range(self.n_hunter)]
        # Initialize Red Banks
        self.pos_red_banks = [
            [randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1), randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)] for _ in
            range(self.n_red_bank)]
        self.maze_red_banks = np.zeros([self.num_grid, self.num_grid])
        for pos_red_bank in self.pos_red_banks:
            x, y = pos_red_bank
            self.maze_red_banks[x, y] = self.maze_red_banks[x, y] + 1
        # Initialize Blue Banks
        self.pos_blue_banks = [
            [randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1), randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)] for _ in
            range(self.n_blue_bank)]
        self.maze_blue_banks = np.zeros([self.num_grid, self.num_grid])
        for pos_blue_bank in self.pos_blue_banks:
            x, y = pos_blue_bank
            self.maze_blue_banks[x, y] = self.maze_blue_banks[x, y] + 1
        # Initialize Red Treasures
        self.pos_red_resources = [
            [randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1), randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)] for _ in
            range(self.n_red_resource)]
        self.maze_red_resources = np.zeros([self.num_grid, self.num_grid])
        for pos_red_resource in self.pos_red_resources:
            x, y = pos_red_resource
            self.maze_red_resources[x, y] = self.maze_red_resources[x, y] + 1
        self.red_treasure_storage_in_resource = [randint(100, 120) for _ in range(self.n_red_resource)]
        # Initialize Blue Resources
        self.pos_blue_resources = [
            [randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1), randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)] for _ in
            range(self.n_blue_resource)]
        self.maze_blue_resources = np.zeros([self.num_grid, self.num_grid])
        for pos_blue_resource in self.pos_blue_resources:
            x, y = pos_blue_resource
            self.maze_blue_resources[x, y] = self.maze_blue_resources[x, y] + 1
        self.blue_treasure_storage_in_resource = [randint(100, 120) for _ in range(self.n_blue_resource)]
        # Initialize Red & Blue Treasure Map
        self.maze_red_treasures = np.zeros([self.num_grid, self.num_grid])
        self.maze_blue_treasures = np.zeros([self.num_grid, self.num_grid])
        return self.get_obs(), self.get_adj()

    def build_env(self):
        # Create the grid map, the border is set to -1.
        maze = np.zeros((self.num_grid, self.num_grid))
        maze[:OBS_RANGE] = -1
        maze[-OBS_RANGE:] = -1
        maze[:, :OBS_RANGE] = -1
        maze[:, -OBS_RANGE:] = -1
        return maze

    def get_obs(self):

        # get observation for hunters
        obs_hunters = []
        # calculate the map of collected treasure for the bank's observation
        maze_collected_red_treasures = np.zeros([self.num_grid, self.num_grid])
        maze_collected_blue_treasures = np.zeros([self.num_grid, self.num_grid])
        for i in range(self.n_hunter):
            obs_hunter_i = []
            x, y = self.pos_hunters[i]
            # binary encoding of position, 14 bit
            binary_x, binary_y = decimal2binary_lst(x, length=7), decimal2binary_lst(x, length=7)
            obs_hunter_i += binary_x
            obs_hunter_i += binary_y
            # stacked feature maps in observation range, (7*7)*7 = 343 bit
            obs_local = hunter_see_local(x, y, self.maze_hunters, self.maze_red_banks, self.maze_blue_banks,
                                         self.maze_red_resources, self.maze_blue_resources, self.maze_red_treasures,
                                         self.maze_blue_treasures)
            obs_hunter_i += obs_local
            # currently hold treasure, 2 bit
            num_hold_red_treasures = self.num_red_treasure_keep_by_hunters[i]
            num_hold_blue_treasures = self.num_blue_treasure_keep_by_hunters[i]
            obs_hunter_i.append(num_hold_red_treasures)
            obs_hunter_i.append(num_hold_blue_treasures)
            maze_collected_red_treasures[x, y] += num_hold_red_treasures
            maze_collected_blue_treasures[x, y] += num_hold_blue_treasures

            # the obs_hunter_i has 14+343+2 = 359 bit
            obs_hunters.append(obs_hunter_i)
        # obs_hunters: [n_hunters, 191]
        obs_hunters = np.array(obs_hunters, dtype=np.float32)
        assert obs_hunters.shape == (self.n_hunter, self.dim_hunter_obs)

        # Get the Red bank observation
        obs_red_banks = []
        for i in range(self.n_red_bank):
            obs_red_bank_i = []
            x, y = self.pos_red_banks[i]
            # binary encoding of position, 14 bit
            binary_x, binary_y = decimal2binary_lst(x, length=7), decimal2binary_lst(x, length=7)
            obs_red_bank_i += binary_x
            obs_red_bank_i += binary_y
            # stacked feature maps in observation range, (7*7)*5 = 245 bit
            obs_local = bank_see_local(x, y, self.maze_hunters, self.maze_red_banks,
                                       self.maze_red_resources, self.maze_red_treasures, maze_collected_red_treasures)
            obs_red_bank_i += obs_local
            # TODO: Should I observe the number of collected treasures in my range?
            # the obs_hunter_i has 14+245 = 259 bit
            obs_red_banks.append(obs_red_bank_i)
        obs_red_banks = np.array(obs_red_banks, dtype=np.float32)
        assert obs_red_banks.shape == (self.n_red_bank, self.dim_bank_obs)

        obs_blue_banks = []
        for i in range(self.n_blue_bank):
            obs_blue_bank_i = []
            x, y = self.pos_blue_banks[i]
            binary_x, binary_y = decimal2binary_lst(x, length=7), decimal2binary_lst(x, length=7)
            obs_blue_bank_i += binary_x
            obs_blue_bank_i += binary_y
            obs_local = bank_see_local(x, y, self.maze_hunters, self.maze_blue_banks,
                                       self.maze_blue_resources, self.maze_blue_treasures,
                                       maze_collected_blue_treasures)
            obs_blue_bank_i += obs_local
            # TODO: Should I observe the number of collected treasures in my range?
            obs_blue_banks.append(obs_blue_bank_i)
        obs_blue_banks = np.array(obs_blue_banks, dtype=np.float32)
        assert obs_blue_banks.shape == (self.n_blue_bank, self.dim_bank_obs)

        obs = {
            'obs_0': obs_hunters,
            'obs_1': obs_red_banks,
            'obs_2': obs_blue_banks
        }
        return obs

    def get_adj(self):
        pos_hunters = np.array(self.pos_hunters)
        pos_red_banks = np.array(self.pos_red_banks)
        pos_blue_banks = np.array(self.pos_blue_banks)

        adj00 = cal_adj(pos_hunters, pos_hunters, np.zeros([self.n_hunter, self.n_hunter], dtype=np.float32))
        adj01 = cal_adj(pos_hunters, pos_red_banks, np.zeros([self.n_hunter, self.n_red_bank], dtype=np.float32))
        adj02 = cal_adj(pos_hunters, pos_blue_banks, np.zeros([self.n_hunter, self.n_blue_bank], dtype=np.float32))
        adj10 = adj01.T
        adj11 = cal_adj(pos_red_banks, pos_red_banks, np.zeros([self.n_red_bank, self.n_red_bank], dtype=np.float32))
        adj12 = cal_adj(pos_red_banks, pos_blue_banks, np.zeros([self.n_red_bank, self.n_blue_bank], dtype=np.float32))
        adj20 = adj02.T
        adj21 = adj12.T
        adj22 = cal_adj(pos_blue_banks, pos_blue_banks,
                        np.zeros([self.n_blue_bank, self.n_blue_bank], dtype=np.float32))
        adj = {
            'adj_0_0': adj00, 'adj_0_1': adj01, 'adj_0_2': adj02,
            'adj_1_0': adj10, 'adj_1_1': adj11, 'adj_1_2': adj12,
            'adj_2_0': adj20, 'adj_2_1': adj21, 'adj_2_2': adj22,
        }
        return adj

    def step(self, actions):
        actions_hunter = actions['act_0']
        actions_red_bank = actions['act_1']
        actions_blue_bank = actions['act_2']

        rewards_hunter = [-0.01] * self.n_hunter
        rewards_red_bank = [-0.01] * self.n_red_bank
        rewards_blue_bank = [-0.01] * self.n_blue_bank

        # Processing hunters to get rewards for hunters, red banks, and blue banks
        for i in range(self.n_hunter):
            pos_hunter = self.pos_hunters[i]
            x, y = pos_hunter
            action_hunter = actions_hunter[i]
            # execute the action and get the reward
            if action_hunter == 0:
                # move leftwards
                if self.maze[x - 1][y] != -1:
                    self.maze_hunters[x][y] -= 1
                    self.maze_hunters[x - 1][y] += 1
                    self.pos_hunters[i][0] -= 1
            elif action_hunter == 1:
                # move rightwards
                if self.maze[x + 1][y] != -1:
                    self.maze_hunters[x][y] -= 1
                    self.maze_hunters[x + 1][y] += 1
                    self.pos_hunters[i][0] += 1
            elif action_hunter == 2:
                # move upwards
                if self.maze[x][y - 1] != -1:
                    self.maze_hunters[x][y] -= 1
                    self.maze_hunters[x][y - 1] += 1
                    self.pos_hunters[i][1] -= 1
            elif action_hunter == 3:
                # move downwards
                if self.maze[x][y + 1] != -1:
                    self.maze_hunters[x][y] -= 1
                    self.maze_hunters[x][y + 1] += 1
                    self.pos_hunters[i][1] += 1
            elif action_hunter == 4:
                # do nothing
                pass
            elif action_hunter == 5:
                # collect treasure
                num_available_red_treasure = self.maze_red_treasures[x, y]
                num_available_blue_treasure = self.maze_blue_treasures[x, y]
                if num_available_red_treasure + num_available_blue_treasure > 0:
                    num_keep_red_treasure = self.num_red_treasure_keep_by_hunters[i]
                    num_keep_blue_treasure = self.num_blue_treasure_keep_by_hunters[i]
                    num_current_keep_treasures = num_keep_red_treasure + num_keep_blue_treasure
                    num_treasure_i_could_take = self.max_keep_treasure - num_current_keep_treasures
                    if num_treasure_i_could_take > 0:
                        # take red treasure first
                        num_take_red_treasure = min(num_treasure_i_could_take, num_available_red_treasure)
                        num_treasure_i_could_take -= num_take_red_treasure
                        self.num_red_treasure_keep_by_hunters[i] += num_take_red_treasure
                        self.maze_red_treasures[x, y] -= num_take_red_treasure
                        num_take_blue_treasure = min(num_treasure_i_could_take, num_available_blue_treasure)
                        num_treasure_i_could_take -= num_take_blue_treasure
                        self.num_blue_treasure_keep_by_hunters[i] += num_take_blue_treasure
                        self.maze_blue_treasures[x, y] -= num_take_blue_treasure
                        # The hunter get small reward for collecting treasures
                        rewards_hunter[i] += self.reward_factor_for_collect_a_treasure * (
                                num_take_red_treasure + num_take_blue_treasure)
            elif action_hunter == 6:
                # dump treasure
                num_red_treasure_i_hold = self.num_red_treasure_keep_by_hunters[i]
                num_blue_treasure_i_hold = self.num_blue_treasure_keep_by_hunters[i]
                if num_red_treasure_i_hold > 0:
                    # i could dump the treasure if there is a red bank in adjacent aera
                    pos_nearby_red_banks = []
                    for x_offset in [-1, 0, 1]:
                        for y_offset in [-1, 0, 1]:
                            if self.maze_red_banks[x + x_offset, y + y_offset] > 0:
                                pos_nearby_red_banks.append([x + x_offset, y + y_offset])
                    if len(pos_nearby_red_banks) > 0:
                        # dump the treasure to nearby red banks equally.
                        # get the id of each nearby bank first since we need to give them reward
                        id_nearby_red_banks = []
                        for id, id_pos_red_bank in enumerate(self.pos_red_banks):
                            if id_pos_red_bank in pos_nearby_red_banks:
                                id_nearby_red_banks.append(id)
                        num_dump_red_treasure = num_red_treasure_i_hold
                        self.num_red_treasure_keep_by_hunters[i] = 0
                        rewards_hunter[i] += self.reward_factor_for_dump_a_treasure * num_dump_red_treasure
                        reward_for_each_bank = self.reward_factor_for_dump_a_treasure * num_dump_red_treasure / len(
                            id_nearby_red_banks)
                        for id_red_bank in id_nearby_red_banks:
                            rewards_red_bank[id_red_bank] += reward_for_each_bank
                if num_blue_treasure_i_hold > 0:
                    # i could dump the treasure if there is a blue bank in adjacent aera
                    pos_nearby_blue_banks = []
                    for x_offset in [-1, 0, 1]:
                        for y_offset in [-1, 0, 1]:
                            if self.maze_blue_banks[x + x_offset, y + y_offset] > 0:
                                pos_nearby_blue_banks.append([x + x_offset, y + y_offset])
                    if len(pos_nearby_blue_banks) > 0:
                        # dump the treasure to nearby blue banks equally.
                        # get the id of each nearby bank first since we need to give them reward
                        id_nearby_blue_banks = []
                        for id, id_pos_blue_bank in enumerate(self.pos_blue_banks):
                            if id_pos_blue_bank in pos_nearby_blue_banks:
                                id_nearby_blue_banks.append(id)
                        num_dump_blue_treasure = num_blue_treasure_i_hold
                        self.num_blue_treasure_keep_by_hunters[i] = 0
                        rewards_hunter[i] += self.reward_factor_for_dump_a_treasure * num_dump_blue_treasure
                        reward_for_each_bank = self.reward_factor_for_dump_a_treasure * num_dump_blue_treasure / len(
                            id_nearby_blue_banks)
                        for id_blue_bank in id_nearby_blue_banks:
                            rewards_blue_bank[id_blue_bank] += reward_for_each_bank
            else:
                raise ValueError("The action value of hunter should in [1-7]")

        # Process Red Bank
        for i in range(self.n_red_bank):
            pos_red_bank = self.pos_red_banks[i]
            x, y = pos_red_bank
            action_red_bank = actions_red_bank[i]
            if action_red_bank == 0:
                # move leftwards
                if self.maze[x - 1][y] != -1:
                    self.maze_red_banks[x][y] -= 1
                    self.maze_red_banks[x - 1][y] += 1
                    self.pos_red_banks[i][0] -= 1
            elif action_red_bank == 1:
                # move rightwards
                if self.maze[x + 1][y] != -1:
                    self.maze_red_banks[x][y] -= 1
                    self.maze_red_banks[x + 1][y] += 1
                    self.pos_red_banks[i][0] += 1
            elif action_red_bank == 2:
                # move upwards
                if self.maze[x][y - 1] != -1:
                    self.maze_red_banks[x][y] -= 1
                    self.maze_red_banks[x][y - 1] += 1
                    self.pos_red_banks[i][1] -= 1
            elif action_red_bank == 3:
                # move downwards
                if self.maze[x][y + 1] != -1:
                    self.maze_red_banks[x][y] -= 1
                    self.maze_red_banks[x][y + 1] += 1
                    self.pos_red_banks[i][1] += 1
            elif action_red_bank == 4:
                # do nothing
                pass
            # TODO: should it get reward when hunter deposit?

        # Process blue Bank
        for i in range(self.n_blue_bank):
            pos_blue_bank = self.pos_blue_banks[i]
            x, y = pos_blue_bank
            action_blue_bank = actions_blue_bank[i]
            if action_blue_bank == 0:
                # move leftwards
                if self.maze[x - 1][y] != -1:
                    self.maze_blue_banks[x][y] -= 1
                    self.maze_blue_banks[x - 1][y] += 1
                    self.pos_blue_banks[i][0] -= 1
            elif action_blue_bank == 1:
                # move rightwards
                if self.maze[x + 1][y] != -1:
                    self.maze_blue_banks[x][y] -= 1
                    self.maze_blue_banks[x + 1][y] += 1
                    self.pos_blue_banks[i][0] += 1
            elif action_blue_bank == 2:
                # move upwards
                if self.maze[x][y - 1] != -1:
                    self.maze_blue_banks[x][y] -= 1
                    self.maze_blue_banks[x][y - 1] += 1
                    self.pos_blue_banks[i][1] -= 1
            elif action_blue_bank == 3:
                # move downwards
                if self.maze[x][y + 1] != -1:
                    self.maze_blue_banks[x][y] -= 1
                    self.maze_blue_banks[x][y + 1] += 1
                    self.pos_blue_banks[i][1] += 1
            elif action_blue_bank == 4:
                # do nothing
                pass

        # Process Red Resource to scatter treasures
        if self.maze_red_treasures.sum() < self.n_red_treasure_capacity:
            for i in range(self.n_red_resource):
                # scatter treasure randomly
                pos_red_resource = self.pos_red_resources[i]
                x_red_resource, y_red_resource = pos_red_resource
                x_red_treasure = x_red_resource + randint(-self.scatter_range, self.scatter_range)
                y_red_treasure = y_red_resource + randint(-self.scatter_range, self.scatter_range)
                if is_legal(x_red_treasure, y_red_treasure):
                    num_treasure = randint(1, 3)
                    self.maze_red_treasures[x_red_treasure][y_red_treasure] += num_treasure
                    self.maze_red_treasures[x_red_treasure][y_red_treasure] \
                        = min(5, self.maze_red_treasures[x_red_treasure][y_red_treasure])
                    self.red_treasure_storage_in_resource[i] -= num_treasure

                    # Refresh a new resource if the treasure storage is empty
                    if self.red_treasure_storage_in_resource[i] <= 0:
                        self.maze_red_resources[x_red_resource, y_red_resource] -= 1
                        self.pos_red_resources[i][0] = randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)
                        self.pos_red_resources[i][1] = randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)
                        self.red_treasure_storage_in_resource[i] = randint(100, 120)
                        x_new_red_resource = self.pos_red_resources[i][0]
                        y_new_red_resource = self.pos_red_resources[i][1]
                        self.maze_red_resources[x_new_red_resource, y_new_red_resource] += 1

        # Process blue Resource to scatter treasures
        if self.maze_blue_treasures.sum() < self.n_blue_treasure_capacity:
            for i in range(self.n_blue_resource):
                # scatter treasure randomly
                pos_blue_resource = self.pos_blue_resources[i]
                x_blue_resource, y_blue_resource = pos_blue_resource
                x_blue_treasure = x_blue_resource + randint(-self.scatter_range, self.scatter_range)
                y_blue_treasure = y_blue_resource + randint(-self.scatter_range, self.scatter_range)
                if is_legal(x_blue_treasure, y_blue_treasure):
                    num_treasure = randint(1, 3)
                    self.maze_blue_treasures[x_blue_treasure][y_blue_treasure] += num_treasure
                    self.maze_blue_treasures[x_blue_treasure][y_blue_treasure] \
                        = min(5, self.maze_blue_treasures[x_blue_treasure][y_blue_treasure])
                    self.blue_treasure_storage_in_resource[i] -= num_treasure

                    # Refresh a new resource if the treasure storage is empty
                    if self.blue_treasure_storage_in_resource[i] <= 0:
                        self.maze_blue_resources[x_blue_resource, y_blue_resource] -= 1
                        self.pos_blue_resources[i][0] = randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)
                        self.pos_blue_resources[i][1] = randint(OBS_RANGE, NUM_GRIDS - OBS_RANGE - 1)
                        self.blue_treasure_storage_in_resource[i] = randint(100, 120)
                        x_new_blue_resource = self.pos_blue_resources[i][0]
                        y_new_blue_resource = self.pos_blue_resources[i][1]
                        self.maze_blue_resources[x_new_blue_resource, y_new_blue_resource] += 1

        next_obs = self.get_obs()
        next_adj = self.get_adj()
        rew = {
            'rew_0': np.array(rewards_hunter, dtype=np.float32),
            'rew_1': np.array(rewards_red_bank, dtype=np.float32),
            'rew_2': np.array(rewards_blue_bank, dtype=np.float32)
        }
        done = {
            'done_0': np.zeros([self.n_hunter, ]),
            'done_1': np.zeros([self.n_red_bank, ]),
            'done_2': np.zeros([self.n_blue_bank, ])
        }
        return rew, next_obs, next_adj, done


if __name__ == '__main__':
    env = CooperativeTreasureCollection()
    env.reset()
    for t in range(10000):
        action0 = [randint(0, 6) for _ in range(env.n_hunter)]
        action1 = [randint(0, 4) for _ in range(env.n_red_bank)]
        action2 = [randint(0, 4) for _ in range(env.n_blue_bank)]
        actions = {'act_0': action0, 'act_1': action1, 'act_2': action2}
        obs, adj, rew, done = env.step(actions)
