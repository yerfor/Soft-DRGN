import numba
import numpy as np
from random import randint
from math import log2, ceil

from utils.hparams import hparams
from scenarios.discrete_grid_base.render_utils import GridRenderer
from scenarios.base_env import BaseScenario

NUM_GRID = hparams['env_num_grid']
NUM_AGENT = hparams['env_num_agent']
OBS_RANGE = hparams['env_obs_range']
COMM_RANGE = hparams['env_comm_range']

X_MIN = Y_MIN = OBS_RANGE
X_MAX = Y_MAX = NUM_GRID - OBS_RANGE - 1

BINARY_POSITION_ENCODING_LENGTH = ceil(log2(NUM_GRID))
BINARY_AGENT_ENCODING_LENGTH = ceil(log2(NUM_AGENT))


@numba.njit
def is_legal(x, y):
    return (x >= X_MIN) and (x <= X_MAX) and (y >= Y_MIN) and (y <= Y_MAX)


@numba.njit
def decimal2binary_lst(x, length=5):
    h = []
    for _ in range(length):  # int类型的位置信息的二进制编码
        h.append(x % 2)
        x = x // 2
    return h


@numba.njit
def see_local(x_t, y_t, maze, maze_ant):
    h = []
    # obs域内有无食物：9位
    for i in range(-OBS_RANGE, OBS_RANGE + 1):  # obs range
        for j in range(-OBS_RANGE, OBS_RANGE + 1):
            h.append(maze[x_t + i][y_t + j])  #

    # obs域内有无其他智能体：9位
    for i in range(-OBS_RANGE, OBS_RANGE + 1):
        for j in range(-OBS_RANGE, OBS_RANGE + 1):
            h.append(maze_ant[x_t + i][y_t + j])
    return h


@numba.njit
def cal_adj(ants, maze_ant, adj):
    n_agent = ants.shape[0]
    for index in range(n_agent):
        x, y = ants[index]
        maze_ant[x][y] = index

    for index in range(n_agent):
        x, y = ants[index]
        for i in range(-COMM_RANGE, COMM_RANGE + 1):  # communication range
            for j in range(-COMM_RANGE, COMM_RANGE + 1):
                # 判断有没有越出网格,如果在这个网格里有agent
                if is_legal(x + i, y + j) and (maze_ant[x + i][y + j] != -1):
                    # 给这个位置的邻居节点置为1
                    adj[index][maze_ant[x + i][y + j]] = 1
    return adj


class Surviving(BaseScenario):
    def __init__(self):
        super(Surviving, self).__init__()
        self.num_grid = NUM_GRID
        self.n_agent = NUM_AGENT
        self.act_dim = 5
        self.max_food = hparams['env_max_food']
        self.capability = 2 * self.n_agent

        self.maze = self._build_maze()
        self.ants = []
        for i in range(self.n_agent):
            self.ants.append([randint(X_MIN, X_MAX), randint(X_MIN, X_MAX)])
        self.maze_ant = None

        self.foods = []
        for i in range(self.n_agent):
            self.foods.append(self.max_food)

        self.n_resource = 8
        self.resource = []
        self.resource_pos = []
        for i in range(self.n_resource):
            self.resource_pos.append([randint(X_MIN, X_MAX), randint(Y_MIN, Y_MAX)])
            self.resource.append(randint(100, 120))
        self.maze_resource_point = None

        self.steps = 0
        self.obs_dim = None
        self.reset()
        self.renderer = None

    def get_mazes(self):
        maze_food = self.maze
        maze_resource = self.maze_resource_point
        maze_agent = self.maze_ant
        return maze_food, maze_resource, maze_agent

    def render(self):
        if self.renderer is None:
            self.renderer = GridRenderer(NUM_GRID)
            self.renderer.init_window()
        mazes = self.get_mazes()
        self.renderer.render(*mazes)

    def init_from_map(self, food_map, agent_map, resource_map=None):
        self.reset()
        self.maze = food_map
        self.maze_ant = agent_map
        self.maze_resource_point = np.zeros((NUM_GRID, NUM_GRID))
        self.poi_pos = []
        self.resource_pos = []
        self.ants = []
        agent0_pos = None
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                num_ant = int(self.maze_ant[x, y])
                if num_ant == 999:  # secret number to specify the agent 0.
                    agent0_pos = [x, y]
                    num_ant = 0
                for _ in range(num_ant):
                    self.ants.append([x, y])
                if resource_map:
                    self.resource_pos.append([x, y])
        if agent0_pos is not None:
            self.ants.insert(0, agent0_pos)
        self.n_agent = len(self.ants)
        self.n_resource = len(self.resource_pos)
        return self.get_obs(), self.get_adj()

    def _build_maze(self):
        # Create the grid map for representing food and boundary, the border is set to -1.
        maze = np.ones((NUM_GRID, NUM_GRID), dtype=np.float32) * -1
        maze[X_MIN:X_MAX + 1, Y_MIN:Y_MAX + 1] = 0
        return maze

    def _reset_the_world(self):
        self.maze = self._build_maze()
        self.ants = []
        for i in range(self.n_agent):  # 随机分配agent的位置
            self.ants.append([randint(X_MIN, X_MAX), randint(Y_MIN, Y_MAX)])
        self.maze_ant = np.zeros((NUM_GRID, NUM_GRID), dtype=np.float32)
        for index in range(self.n_agent):
            x = self.ants[index][0]
            y = self.ants[index][1]
            self.maze_ant[x][y] += 1.
        self.foods = []  # 给每个agent配备满食物
        for i in range(self.n_agent):
            self.foods.append(self.max_food)

        self.resource = []  # 食物点还有多少储备
        self.resource_pos = []  # 食物点的位置
        for i in range(self.n_resource):
            self.resource_pos.append([randint(X_MIN, X_MAX), randint(Y_MIN, Y_MAX)])
            self.resource.append(randint(100, 120))
        self.maze_resource_point = np.zeros((NUM_GRID, NUM_GRID), dtype=np.float32)
        for index in range(self.n_resource):
            x = self.resource_pos[index][0]
            y = self.resource_pos[index][1]
            self.maze_resource_point[x][y] += 1

    def reset(self):
        self._reset_the_world()
        obs, adj = self.get_obs(), self.get_adj()
        self.obs_dim = obs.shape[-1]
        return obs, adj

    def get_obs(self):
        obs = []
        maze_ant = np.zeros((NUM_GRID, NUM_GRID), dtype=np.float32)
        for agent_index in range(self.n_agent):
            x = self.ants[agent_index][0]
            y = self.ants[agent_index][1]
            maze_ant[x][y] = 1
        self.maze_ant = maze_ant

        for agent_index in range(self.n_agent):
            h = []  # we use list to store each element of the observation
            x = self.ants[agent_index][0]
            y = self.ants[agent_index][1]

            # 自身位置的encoding：10位
            h1 = decimal2binary_lst(x, length=BINARY_POSITION_ENCODING_LENGTH)
            h2 = decimal2binary_lst(y, length=BINARY_POSITION_ENCODING_LENGTH)
            h = h1 + h2

            # obs域内有无食物：9位,有无其他智能体,9位
            # JIT: 5us==>800ns
            obs_local = see_local(x, y, self.maze, maze_ant)
            h += obs_local

            agent_id_encoding = decimal2binary_lst(agent_index, length=BINARY_AGENT_ENCODING_LENGTH)
            h += agent_id_encoding

            # 本智能体携带的食物数量：1位
            h.append(self.foods[agent_index])

            obs.append(h)
        obs = np.array(obs, dtype=np.float32).squeeze()  # [n_agent,obs_dim]
        return obs

    def get_adj(self):
        maze_ant = np.ones((NUM_GRID, NUM_GRID), dtype=np.int) * -1  # 记录32x32的网格里哪一格有哪个agent
        adj = np.zeros((self.n_agent, self.n_agent))
        ants = np.array(self.ants, dtype=int)
        adj = cal_adj(ants, maze_ant, adj)
        adj = np.array(adj, dtype=np.float32).reshape([self.n_agent, self.n_agent])
        return adj

    def _execute_actions_get_reward(self, actions):
        for i in range(self.n_agent):
            x = self.ants[i][0]
            y = self.ants[i][1]
            # 执行智能体的移动
            if actions[i] == 0:
                if self.maze[x - 1][y] != -1:
                    self.ants[i][0] = x - 1
            if actions[i] == 1:
                if self.maze[x + 1][y] != -1:
                    self.ants[i][0] = x + 1
            if actions[i] == 2:
                if self.maze[x][y - 1] != -1:
                    self.ants[i][1] = y - 1
            if actions[i] == 3:
                if self.maze[x][y + 1] != -1:
                    self.ants[i][1] = y + 1
            if actions[i] == 4:
                self.foods[i] += 2 * self.maze[x][y]
                self.maze[x][y] = 0

            self.foods[i] = max(0, min(self.foods[i] - 1, self.max_food))

        reward = [0.4] * self.n_agent
        for i in range(self.n_agent):
            if self.foods[i] == 0:
                reward[i] = - 0.2
        reward = np.array(reward, dtype=np.float32).reshape([self.n_agent, ])
        return reward

    def get_done(self):
        return np.zeros(shape=[self.n_agent], dtype=np.float32)

    def step(self, actions):
        # actions: list[int], [n_agent,]

        reward = self._execute_actions_get_reward(actions)
        done = self.get_done()

        if (self.maze.sum() + 120) > self.capability:  # 保持地图上的食物总量满足少于某个约束
            return reward, self.get_obs(), self.get_adj(), done

        # if the number of food is smaller than the capacity
        for i in range(self.n_resource):
            # 资源点随机向周围撒食物
            x = self.resource_pos[i][0] + randint(-3, 3)
            y = self.resource_pos[i][1] + randint(-3, 3)
            if is_legal(x, y):
                num = randint(1, 5)
                self.maze[x][y] += num  # 往随机位置随机量的食物
                self.maze[x][y] = min(self.maze[x][y], 5)
                self.resource[i] -= num  # 资源点减少对应量的食物
                if self.resource[i] <= 0:  # 如果本资源点耗尽了，在附近重生一个资源点
                    self.resource_pos[i][0] = randint(X_MIN, X_MAX)
                    self.resource_pos[i][1] = randint(Y_MIN, Y_MAX)
                    self.resource[i] = randint(100, 120)
                    self.maze_resource_point = np.zeros((NUM_GRID, NUM_GRID))
                    for index in range(self.n_resource):
                        x = self.resource_pos[index][0]
                        y = self.resource_pos[index][1]
                        self.maze_resource_point[x][y] = 1

        return reward, self.get_obs(), self.get_adj(), done
