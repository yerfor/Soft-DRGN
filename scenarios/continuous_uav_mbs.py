import math
import random

import networkx as nx
import numba
import numpy as np
import pygame.image
import copy
from sko.GA import GA

from scenarios.continuous_uav_base.agents import AgentGroup2D
from utils.hparams import hparams
from scenarios.continuous_uav_base.render_utils import ContinuousWorldRenderer

N_AGENT = hparams['env_num_agent']
N_POI = hparams['env_num_poi']
N_MAJOR = hparams['env_num_major_point']
MAJOR_POINT_MIN_DISTANCE_PERCENT = hparams['env_major_point_min_distant_percent']
N_OBSTACLE_RECT = hparams['env_num_rect_obstacle']
N_OBSTACLE_CIRCLE = hparams['env_num_circle_obstacle']
NUM_GRIDS = hparams['env_num_grid']
COVER_RANGE = hparams['env_coverage_range']
OBS_RANGE = hparams['env_observe_range']
COMM_RANGE = hparams['env_comm_range']
COLLIDE_RANGE = hparams['env_collide_range']

# non-pixel configs
MAX_NEIGHBOR_AGENTS = hparams['env_num_max_neighbor_agent']
MAX_NEIGHBOR_POI = hparams['env_num_max_neighbor_poi']

SQUARE_OBS = hparams['env_use_square_obs']
PIXEL_OBS = hparams['env_use_pixel_obs']
STATIC_POI = hparams['env_static_poi']
ENABLE_OBSTACLE = hparams['env_enable_obstacle']


def see_circle(x, y, x_offsets, y_offsets, map0, map1, map2):
    map0 = map0[x + x_offsets, y + y_offsets]
    map1 = map1[x + x_offsets, y + y_offsets]
    map2 = map2[x + x_offsets, y + y_offsets]
    return np.concatenate([map0, map1, map2])


def see_local(x, y, agent_map, poi_map, obstacle_map, square=SQUARE_OBS, local_indices=None):
    if square:
        raw_agent_obs_map = agent_map[x - OBS_RANGE:x + OBS_RANGE + 1, y - OBS_RANGE:y + OBS_RANGE + 1]
        reshape_agent_obs_map = raw_agent_obs_map.reshape(
            [3, (2 * OBS_RANGE + 1) // 3, (2 * OBS_RANGE + 1) // 3, 3])
        reduced_agent_obs_map = reshape_agent_obs_map.sum(axis=0).sum(axis=-1).reshape([-1])

        raw_poi_obs_map = poi_map[x - OBS_RANGE:x + OBS_RANGE + 1, y - OBS_RANGE:y + OBS_RANGE + 1]
        reshape_poi_obs_map = raw_poi_obs_map.reshape([3, (2 * OBS_RANGE + 1) // 3, (2 * OBS_RANGE + 1) // 3, 3])
        reduced_poi_obs_map = reshape_poi_obs_map.sum(axis=0).sum(axis=-1).reshape([-1])

        raw_obstacle_obs_map = obstacle_map[x - OBS_RANGE:x + OBS_RANGE + 1, y - OBS_RANGE:y + OBS_RANGE + 1]
        reshape_obstacle_obs_map = raw_obstacle_obs_map.reshape(
            [3, (2 * OBS_RANGE + 1) // 3, (2 * OBS_RANGE + 1) // 3, 3])
        reduced_obstacle_obs_map = reshape_obstacle_obs_map.sum(axis=0).sum(axis=-1).reshape([-1])

        return np.concatenate([reduced_agent_obs_map, reduced_poi_obs_map, reduced_obstacle_obs_map])
    else:
        local_obs = see_circle(x, y, local_indices[0], local_indices[1], agent_map, poi_map, obstacle_map)
        return local_obs


@numba.njit
def cal_two_point_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@numba.njit
def decimal2binary_arr(x, length=10):
    h = []
    for _ in range(length):  # int类型的位置信息的二进制编码
        h.append(x % 2)
        x = x // 2
    return np.array(h)


# @numba.njit
def cal_cover_num(poi_pos, group, agent_pos, i):
    individual_covered = 0
    total_covered = 0
    for j, (poi_x, poi_y) in enumerate(poi_pos):
        covered_by_i = False
        covered_by_others = False
        for agent_id in group:
            agent_x, agent_y = agent_pos[agent_id]
            if cal_two_point_distance(agent_x, agent_y, poi_x, poi_y) < COVER_RANGE:
                if i == agent_id:
                    covered_by_i = True
                else:
                    covered_by_others = True
        if covered_by_i and not covered_by_others:
            individual_covered += 1
        if covered_by_i or covered_by_others:
            total_covered += 1
    return individual_covered, total_covered


@numba.njit
def cal_poi_is_covered(poi_is_covered, poi_pos, agent_pos):
    for i, (poi_x, poi_y) in enumerate(poi_pos):
        for j, (agent_x, agent_y) in enumerate(agent_pos):
            if cal_two_point_distance(poi_x, poi_y, agent_x, agent_y) <= COVER_RANGE:
                poi_is_covered[i] = 1
                break
    return poi_is_covered


@numba.njit
def in_rectangle(x, y, min_x, min_y, height, width):
    return x > min_x and x < min_x + height and y > min_y and y < min_y + width


@numba.njit
def in_circle(x, y, x_core, y_core, radius):
    distance_to_core = ((x - x_core) ** 2 + (y - y_core) ** 2) ** 0.5
    return distance_to_core < radius


@numba.njit
def cal_obstacle_map(obstacle_map, rect_obstacles, circle_obstacles):
    world_scale = obstacle_map.shape
    for x in range(world_scale[0]):
        for y in range(world_scale[1]):
            if x < OBS_RANGE or x > NUM_GRIDS + OBS_RANGE:
                obstacle_map[x, y] = 1
                continue
            if y < OBS_RANGE or y > NUM_GRIDS + OBS_RANGE:
                obstacle_map[x, y] = 1
                continue
            for (x_min, y_min, height, width) in rect_obstacles:
                if in_rectangle(x, y, x_min, y_min, height, width):
                    obstacle_map[x, y] = 1
                    break
            for (x_core, y_core, radius) in circle_obstacles:
                if in_circle(x, y, x_core, y_core, radius):
                    obstacle_map[x, y] = 1
                    break
    return obstacle_map


@numba.njit
def generate_n_agent_not_in_map(n, map):
    out = []
    for _ in range(n):
        while True:
            x = random.random() * 0.8 + 0.1
            y = random.random() * 0.8 + 0.1
            x_idx = int(x * NUM_GRIDS + OBS_RANGE)
            y_idx = int(y * NUM_GRIDS + OBS_RANGE)
            if map[x_idx, y_idx] == 0:
                out.append([x, y])
                break
    return np.array(out, dtype=np.float32)


@numba.njit
def clip(x, x_min, x_max):
    if x > x_max:
        return x_max
    if x < x_min:
        return x_min
    return x


@numba.njit
def generate_poi_not_in_map(n_major, n_poi, map):
    pos_major = []
    for i in range(n_major):
        while True:
            x = random.random() * 0.7 + 0.15
            y = random.random() * 0.7 + 0.15
            x_idx = int(x * NUM_GRIDS + OBS_RANGE)
            y_idx = int(y * NUM_GRIDS + OBS_RANGE)
            if map[x_idx, y_idx] == 0:
                if len(pos_major) > 0:
                    not_good = False
                    for pos in pos_major:
                        if cal_two_point_distance(x, y, pos[0], pos[1]) < MAJOR_POINT_MIN_DISTANCE_PERCENT:
                            not_good = True
                            break
                    if not_good:
                        continue
                    pos_major.append([x, y])
                    break
                else:
                    pos_major.append([x, y])
                    break
    pos_major = np.array(pos_major)
    poi_pos = []
    for i in range(n_major):
        num_scattering_poi = n_poi // n_major
        pos_major_i = pos_major[i]
        x_major_i, y_major_i = pos_major_i
        for j in range(num_scattering_poi):
            while True:
                x_diff = np.random.randn(1)[0] * 0.1
                x_poi = clip(x_major_i + x_diff, 0, 1)

                # y_diff = (random.random() - 0.5) * 2 * 0.3
                y_diff = np.random.randn(1)[0] * 0.1
                y_poi = clip(y_major_i + y_diff, 0, 1)
                x_index = int(x_poi * NUM_GRIDS + OBS_RANGE)
                y_index = int(y_poi * NUM_GRIDS + OBS_RANGE)
                if map[x_index, y_index] == 0:
                    poi_pos.append([x_poi, y_poi])
                    break
    poi_pos = np.array(poi_pos, dtype=np.float32)
    return pos_major, poi_pos


class AgentUAVMBS(AgentGroup2D):
    def __init__(self, n_agent, pos=None):
        self.mass = 1
        super(AgentUAVMBS, self).__init__(n_agent, pos=pos, vel=None, mass=self.mass)
        self.n_agent = n_agent
        self.max_force = 0.02  # 1m/s^2
        self.max_vel = 0.02  # 10m/s
        self.hover_energy_cost = 0.5
        self.move_energy_cost = 0.5
        self.energy_consumption = np.ones([n_agent]) * self.hover_energy_cost
        self.action2force = self.max_force * np.array([
            [1, 0],  # action 0 denotes 1 unit force downward
            [math.sqrt(2) / 2, -math.sqrt(2) / 2],  # left down side
            [0, -1],  # leftward
            [-math.sqrt(2) / 2, -math.sqrt(2) / 2],  # left upwards
            [-1, 0],  # upward
            [-math.sqrt(2) / 2, math.sqrt(2) / 2],  # right upward
            [0, 1],  # rightward
            [math.sqrt(2) / 2, math.sqrt(2) / 2],  # right downward

            [0.5, 0],  # action 0 denotes 1 unit force downward
            [math.sqrt(2) / 4, -math.sqrt(2) / 4],  # left down side
            [0, -0.5],  # leftward
            [-math.sqrt(2) / 4, -math.sqrt(2) / 4],  # left upwards
            [-0.5, 0],  # upward
            [-math.sqrt(2) / 4, math.sqrt(2) / 4],  # right upward
            [0, 0.5],  # rightward
            [math.sqrt(2) / 4, math.sqrt(2) / 4],  # right downward
            [0, 0]  # no throttle
        ])
        self.n_action = self.action2force.shape[0]

    def deploy_action(self, actions, obstacle_map, time_step=1):
        assert isinstance(actions, np.ndarray)
        actions = actions.reshape([-1]).astype(int)
        assert actions.size == self.n_agent
        force = self.action2force[actions]  # [n_agent, 2]
        acceleration = force / self.mass
        self.vel = self.vel + acceleration * time_step
        vel_norm = (self.vel ** 2).sum(axis=-1, keepdims=True) ** 0.5
        vel_less_than_threshold_mask = np.array(vel_norm < self.max_vel, dtype=np.float32)
        vel_greater_than_threshold_mask = np.array(vel_norm > self.max_vel, dtype=np.float32)
        vel_greater_than_threshold_mask = vel_greater_than_threshold_mask * (
                self.max_vel * np.ones_like(vel_norm) / (vel_norm + 1e-5))
        clip_vel_mask = vel_less_than_threshold_mask + vel_greater_than_threshold_mask
        self.vel = self.vel * clip_vel_mask
        clipped_vel_norm = (self.vel ** 2).sum(axis=-1, keepdims=True) ** 0.5
        self.energy_consumption = np.ones([self.n_agent]) * self.hover_energy_cost + (
                clipped_vel_norm / self.max_vel * self.move_energy_cost).reshape([-1])

        collision_penalty = np.zeros([self.n_agent], dtype=np.float32)
        previous_pos = self.pos
        self.pos = self.pos + self.vel * time_step
        self.pos = np.clip(self.pos, 0, 1)
        for i, (x_i, y_i) in enumerate(self.pos):
            x_i = int(x_i * NUM_GRIDS + OBS_RANGE)
            y_i = int(y_i * NUM_GRIDS + OBS_RANGE)
            if obstacle_map[x_i, y_i] != 0:
                self.pos[i] = previous_pos[i]
                collision_penalty[i] = -1
        return collision_penalty


class ContinuousUAVMBS:
    def __init__(self, n_agent=N_AGENT, world_scale=(NUM_GRIDS + 2 * OBS_RANGE, NUM_GRIDS + 2 * OBS_RANGE),
                 heuristic_reward=True):
        self.world_scale = np.array(world_scale, dtype=np.int32)
        self.n_agent = n_agent
        self.pixel_obs = PIXEL_OBS
        self.static_poi = STATIC_POI
        self.enable_obstacle = ENABLE_OBSTACLE
        self.square_obs = SQUARE_OBS
        self.heuristic_reward = heuristic_reward
        self.agents = None
        self.agent_map = np.zeros(world_scale)
        self.num_major_points = N_MAJOR
        self.num_poi = N_POI
        self.major_point_pos = np.zeros([self.num_major_points, 2])
        self.poi_pos = np.zeros([self.num_major_points, 2])
        self.poi_map = np.zeros(world_scale)
        self.num_rect_obstacles = N_OBSTACLE_RECT if ENABLE_OBSTACLE else 0
        self.num_circle_obstacles = N_OBSTACLE_CIRCLE if ENABLE_OBSTACLE else 0
        self.obstacle_max_width = 20
        self.rect_obstacles = np.zeros([self.num_circle_obstacles, 3])  # represent as (x,y,r)
        self.circle_obstacles = np.zeros([self.num_rect_obstacles, 4])  # (x_min,y_min,x_max,y_max)
        self.obstacle_map = np.zeros(world_scale)
        self.local_indices_x = []
        self.local_indices_y = []
        for offset_x in range(-OBS_RANGE, OBS_RANGE):
            for offset_y in range(-OBS_RANGE, OBS_RANGE):
                if (offset_x ** 2 + offset_y ** 2) ** 0.5 <= OBS_RANGE:
                    self.local_indices_x.append(offset_x)
                    self.local_indices_y.append(offset_y)
        self.local_indices_x = np.array(self.local_indices_x)
        self.local_indices_y = np.array(self.local_indices_y)
        if self.pixel_obs:
            if self.square_obs:
                self.obs_dim = 20 + 2 + 3 * ((2 * OBS_RANGE + 1) // 3) ** 2
            else:
                self.obs_dim = 30 + 2 + 3 * len(self.local_indices_x)
        else:
            self.obs_dim = 4 + 4 * MAX_NEIGHBOR_AGENTS + 2 * MAX_NEIGHBOR_POI
        self.act_dim = None

        self.poi_coverage_history = []
        self.energy_consumption_history = []
        self.num_neighbor_history = []
        self.timeslot = 0
        self.reset()
        self.renderer = None

    def render(self):
        if self.renderer is None:
            self.renderer = ContinuousWorldRenderer(num_grid=NUM_GRIDS, obs_range=OBS_RANGE, fps=10, scale_factor=2.5)
            self.agent_surface = pygame.image.load("scenarios/continuous_uav_base/UAV.png")
        render_info = self.get_render_info()
        self.renderer.render_uav(render_info, agent_surface=self.agent_surface)

    def reset(self):
        self.timeslot = 0
        # generate rectangle obstacle
        xy_min = (np.random.rand(self.num_rect_obstacles, 2) * 0.8 + 0.1) * NUM_GRIDS + OBS_RANGE
        height_width = (np.random.rand(self.num_rect_obstacles, 2) + 1) / 2 * self.obstacle_max_width

        self.rect_obstacles = np.concatenate([xy_min, height_width], axis=-1)  # [n_obstacle, 4]

        # generate circle obstacle
        xy = (np.random.rand(self.num_circle_obstacles, 2) * 0.8 + 0.1) * NUM_GRIDS + OBS_RANGE
        radius = (np.random.rand(self.num_circle_obstacles, 1) + 1) / 2 * 0.5 * self.obstacle_max_width
        self.circle_obstacles = np.concatenate([xy, radius], axis=-1)
        self.refresh_obstacle_map()

        pos_major, poi_pos = generate_poi_not_in_map(n_major=self.num_major_points, n_poi=self.num_poi,
                                                     map=self.obstacle_map)
        self.major_point_pos = pos_major
        self.poi_pos = poi_pos
        self.refresh_poi_map()

        agent_pos = generate_n_agent_not_in_map(self.n_agent, self.obstacle_map)
        self.agents = AgentUAVMBS(self.n_agent, agent_pos)
        self.act_dim = self.agents.n_action
        self.refresh_agent_map()

        adj = self.get_adj()
        obs = self.get_obs()

        self.poi_coverage_history = []
        self.energy_consumption_history = []
        self.num_neighbor_history = []
        self.poi_is_covered = cal_poi_is_covered(np.zeros([self.num_poi]), poi_pos, agent_pos)

        return obs, adj

    def init_from_map(self, poi_map, agent_map):
        self.reset()
        self.poi_pos = []
        agent_pos_lst = []
        for grid_x in range(poi_map.shape[0]):
            for grid_y in range(poi_map.shape[1]):
                num_poi = int(poi_map[grid_x][grid_y])
                for _ in range(num_poi):
                    self.poi_pos.append([grid_x / NUM_GRIDS, grid_y / NUM_GRIDS])
        for grid_x in range(agent_map.shape[0]):
            for grid_y in range(agent_map.shape[1]):
                num_agent = int(agent_map[grid_x][grid_y])
                for _ in range(num_agent):
                    agent_pos_lst.append([grid_x / NUM_GRIDS, grid_y / NUM_GRIDS])
        self.num_poi = len(self.poi_pos)
        self.poi_pos = np.array(self.poi_pos)
        agent_pos = np.array(agent_pos_lst).reshape([-1, 2])
        self.n_agent = len(agent_pos)
        self.agents = AgentUAVMBS(n_agent=self.n_agent, pos=agent_pos)
        self.refresh_agent_map()
        self.refresh_poi_map()
        return self.get_obs(), self.get_adj()

    def refresh_poi_map(self):
        self.poi_map = np.zeros(self.world_scale, dtype=np.int32)
        self.x_pois = np.clip(self.poi_pos[:, 0] * NUM_GRIDS + OBS_RANGE, OBS_RANGE, NUM_GRIDS + OBS_RANGE - 1).astype(
            np.int32)
        self.y_pois = np.clip(self.poi_pos[:, 1] * NUM_GRIDS + OBS_RANGE, OBS_RANGE, NUM_GRIDS + OBS_RANGE - 1).astype(
            np.int32)
        for x, y in zip(self.x_pois, self.y_pois):
            self.poi_map[x, y] += 1

    def refresh_obstacle_map(self):
        self.obstacle_map = cal_obstacle_map(np.zeros(self.world_scale, dtype=np.int32),
                                             self.rect_obstacles, self.circle_obstacles)

    def refresh_agent_map(self):
        self.agent_map = np.zeros(self.world_scale, dtype=np.int32)
        self.x_agents = np.clip(self.agents.x * NUM_GRIDS + OBS_RANGE, OBS_RANGE, NUM_GRIDS + OBS_RANGE - 1).astype(
            np.int32)
        self.y_agents = np.clip(self.agents.y * NUM_GRIDS + OBS_RANGE, OBS_RANGE, NUM_GRIDS + OBS_RANGE - 1).astype(
            np.int32)
        for x, y in zip(self.x_agents, self.y_agents):
            self.agent_map[x, y] += 1

    def get_mazes(self):
        if not self.pixel_obs:
            print("warning: you are acquiring agent map in no-pixel-obs mode!")
        return self.poi_map, self.agent_map

    def get_pos(self):
        return self.poi_pos, self.agents.pos, self.rect_obstacles, self.circle_obstacles

    def get_render_info(self):
        render_info = {}
        render_info['poi_pos'] = self.poi_pos
        render_info['agent_pos'] = self.agents.pos
        render_info['rect_obstacles'] = self.rect_obstacles
        render_info['circle_obstacles'] = self.circle_obstacles
        render_info['poi_cover_id'] = self.poi_is_covered
        render_info['poi_cover_percent'] = sum(self.poi_is_covered) / self.num_poi
        c_t, f_t, _, _ = self.cal_episodic_coverage_and_fairness()
        render_info['episode_coverage_item'] = c_t
        render_info['episode_fairness_item'] = f_t
        render_info['energy_consumption'] = self.agents.energy_consumption[-1]
        render_info['timeslot'] = self.timeslot
        return render_info

    def get_obs(self):
        if self.pixel_obs:

            obs = []
            for i, (x, y) in enumerate(zip(self.x_agents, self.y_agents)):
                obs_i = []
                binary_x = decimal2binary_arr(int(x), length=10)
                obs_i.append(binary_x)
                binary_y = decimal2binary_arr(int(y), length=10)
                obs_i.append(binary_y)
                if self.square_obs:
                    h_local = see_local(x, y, self.agent_map, self.poi_map, self.obstacle_map)
                else:
                    h_local = see_local(x, y, self.agent_map, self.poi_map, self.obstacle_map,
                                        local_indices=[self.local_indices_x, self.local_indices_y])
                obs_i.append(h_local)

                vel = self.agents.vel[i] / self.agents.max_vel
                obs_i.append(vel)

                binary_id = np.array(decimal2binary_arr(int(i), length=10))
                obs_i.append(binary_id)

                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)
            obs = np.array(obs)
        else:
            agent_adj = self.adj_t  # [n_agent, n_agent]
            agent_pos = self.agents.pos  # [n_agent, 2]
            agent_vel = self.agents.vel  # [n_agent, 2]
            poi_pos = self.poi_pos  # [n_poi, 2]
            n_poi = self.num_poi
            n_agent = self.n_agent
            normalized_obs_range = OBS_RANGE / NUM_GRIDS
            dummy_value = -3
            n_max_neighbor_agent = MAX_NEIGHBOR_AGENTS
            n_max_neighbor_poi = MAX_NEIGHBOR_POI

            tmp_agent_i_pos = np.repeat(agent_pos[:, np.newaxis, :], repeats=n_agent, axis=1)  # [n_agent, n_poi, 2]
            tmp_agent_j_pos = np.repeat(agent_pos[np.newaxis, :, :], repeats=n_agent, axis=0)  # [n_agent, n_poi, 2]
            agent_diff = tmp_agent_j_pos - tmp_agent_i_pos

            tmp_agent_pos = np.repeat(agent_pos[:, np.newaxis, :], repeats=n_poi, axis=1)  # [n_agent, n_poi, 2]
            tmp_poi_pos = np.repeat(poi_pos[np.newaxis, :], repeats=n_agent, axis=0)  # [n_agent, n_poi, 2]
            agent_poi_diff = tmp_poi_pos - tmp_agent_pos  # [n_agent, n_poi, 2]
            agent_poi_distance = (agent_poi_diff ** 2).sum(axis=-1) ** 0.5  # [n_agent, n_poi]
            agent_poi_adj = (agent_poi_distance < normalized_obs_range).astype(np.float32)

            obs = []
            for i in range(n_agent):
                obs_i = []

                self_pos = agent_pos[i].reshape([-1])  # 2 bit
                self_vel = agent_vel[i].reshape([-1])  # 2 bit
                obs_i.append(self_pos)
                obs_i.append(self_vel)

                # pos and vel of other_agents: 2*10 + 2*10 = 40 bit
                neighboring_agents = np.where(agent_adj[i] > 0)[0]
                # UAV within Comm range is in the obs space!
                n_insert_agent = 0
                for j in neighboring_agents:
                    if j != i and n_insert_agent < n_max_neighbor_agent:
                        other_pos_diff = agent_diff[i, j].reshape([-1])  # 2 bit
                        other_vel = agent_vel[j].reshape([-1])
                        obs_i.append(other_pos_diff)
                        obs_i.append(other_vel)
                        n_insert_agent += 1
                n_agent_padding = max(0, n_max_neighbor_agent - n_insert_agent)
                obs_i.append(np.ones([n_agent_padding * 2, ]) * dummy_value)  # padding agent pos
                obs_i.append(np.ones([n_agent_padding * 2, ]) * 0)  # padding agent vel

                # pos of pois: 2*30 = 60 bit
                neighboring_pois = np.where(agent_poi_adj[i] > 0)[0]
                # POI within OBS range is in the obs space!
                n_insert_poi = 0
                for j in neighboring_pois:
                    if n_insert_poi < n_max_neighbor_poi - 1:
                        other_pos_diff = agent_poi_diff[i, j].reshape([-1])  # 2 bit
                        obs_i.append(other_pos_diff)
                        n_insert_poi += 1
                n_poi_padding = max(0, n_max_neighbor_poi - n_insert_poi)
                obs_i.append(np.ones([n_poi_padding * 2, ]) * dummy_value)  # padding agent pos

                obs_i = np.concatenate(obs_i)  # [104, ]
                obs.append(obs_i)
            obs = np.array(obs)  # [n_agent, 104]
        return obs.reshape([self.n_agent, self.obs_dim])

    def get_adj(self):
        pos_agents = np.concatenate([self.x_agents.reshape([-1, 1]),
                                     self.y_agents.reshape([-1, 1])], axis=-1)  # [n_agent, 2]
        pos_agents_i = pos_agents.reshape([self.n_agent, 1, 2]).repeat(self.n_agent, axis=1)
        pos_agents_j = pos_agents.reshape([1, self.n_agent, 2]).repeat(self.n_agent, axis=0)
        distance_mat = ((pos_agents_i - pos_agents_j) ** 2).sum(axis=-1) ** 0.5  # [n_agent, n_agent]
        adj = (distance_mat <= COMM_RANGE).astype(np.float32)
        collide_adj = (distance_mat <= COVER_RANGE * 0.5).astype(np.float32)
        self.collide_agent_num = collide_adj.sum(axis=1).reshape([-1]) - 1
        self.adj_t = adj
        mean_num_neighbor = (adj.sum() - self.n_agent) / self.n_agent
        self.num_neighbor_history.append(mean_num_neighbor)
        return adj.reshape([self.n_agent, self.n_agent])

    def step(self, actions):
        collision_penalty = self.agents.deploy_action(actions, self.obstacle_map)
        self.energy_consumption_history.append(self.agents.energy_consumption)
        if self.pixel_obs:
            self.refresh_agent_map()
        adj = self.get_adj()

        adj_graph = nx.from_numpy_matrix(adj)
        sub_graphs = tuple(
            adj_graph.subgraph(c).nodes() for c in nx.connected_components(adj_graph))  # ((0,1,2),(3,4,),...)
        reward = []
        id2group_dic = {}
        for i_group, group in enumerate(sub_graphs):
            for id in group:
                id2group_dic[id] = i_group
        poi_pos = np.concatenate([self.x_pois.reshape([self.num_poi, 1]),
                                  self.y_pois.reshape([self.num_poi, 1])], axis=-1)
        agent_pos = np.concatenate([self.x_agents.reshape([-1, 1]),
                                    self.y_agents.reshape([-1, 1])], axis=-1)  # [n_agent, 2]

        self.poi_is_covered = cal_poi_is_covered(np.zeros([self.num_poi]), poi_pos, agent_pos)
        self.poi_coverage_history.append(self.poi_is_covered)

        if self.heuristic_reward:
            for i in range(self.n_agent):
                i_group = id2group_dic[i]
                group = sub_graphs[i_group]  # [1,4,6,...,55,100]

                individual_covered, total_covered = cal_cover_num(poi_pos, tuple(group), agent_pos, i)

                individual_rew = -1 if individual_covered == 0 else 1 * individual_covered
                group_scale = len(group)
                group_rew = 0. if group_scale == 1 else 0.1 * (total_covered - individual_covered) / (group_scale - 1)
                rew = group_rew + individual_rew
                collide_percent = COLLIDE_RANGE / NUM_GRIDS

                # penalty to agents near the boundary
                if not in_rectangle(self.agents.pos[i][0], self.agents.pos[i][1], collide_percent, collide_percent,
                                    1 - collide_percent, 1 - collide_percent):
                    rew -= 1
                # rew -= self.collide_agent_num[i]
                rew /= self.agents.energy_consumption[i]
                reward.append(rew)
        else:
            num_covered = self.poi_is_covered.sum()
            current_c = num_covered / self.num_poi
            f, _, _, _ = self.cal_episodic_coverage_and_fairness()
            global_rew = f * current_c / self.agents.energy_consumption.mean()
            reward = [global_rew] * self.n_agent

        rew = np.array(reward)
        rew = rew + collision_penalty
        rew = rew.reshape([self.n_agent, ])
        done = np.zeros([self.n_agent, ])

        if not self.static_poi:
            self.poi_pos = np.clip(self.poi_pos + 0.001 * (np.random.randn(self.num_poi, 2)), a_min=0, a_max=1)
        obs = self.get_obs()
        self.timeslot += 1
        return rew, obs, adj, done

    def cal_episodic_coverage_and_fairness(self):
        try:
            w_t_k = np.stack(self.poi_coverage_history)  # [T, n_poi]
        except:
            # First time
            return -1, -1, -1, -1
        for i in range(1, len(w_t_k)):
            w_t_k[i] = w_t_k[i] + w_t_k[i - 1]
        max_time = w_t_k.shape[0]
        time_matrix = np.arange(1, max_time + 1).reshape([max_time, 1]).repeat(self.num_poi, axis=-1)  # [T, n_poi]
        c_t_k = w_t_k / time_matrix
        c_t = np.mean(c_t_k, axis=-1)
        final_averaged_coverage_score = c_t[-1]
        f_t = (np.sum(c_t_k, axis=-1) ** 2) / (self.num_poi * np.sum(c_t_k ** 2, axis=-1) + 1e-10)
        final_achieved_fairness_index = f_t[-1]
        return final_averaged_coverage_score, final_achieved_fairness_index, c_t, f_t

    def cal_episodic_mean_energy_consumption(self):
        return np.array(self.energy_consumption_history).mean()

    def cal_episodic_mean_num_neighbors(self):
        return np.array(self.num_neighbor_history).mean()

    def from_copy(self, other):
        for k, v in self.__dict__.items():
            self.__dict__[k] = copy.deepcopy(other.__dict__[k])

    def get_log_vars(self):
        """
        get vars to be logged in a episode
        """
        coverage_index, fairness_index, _, _ = self.cal_episodic_coverage_and_fairness()
        energy_index = self.cal_episodic_mean_energy_consumption()
        return {'coverage_index': coverage_index, 'fairness_index': fairness_index, 'energy_index': energy_index}


def MB_greedy_policy(env):
    # It is a model-based individual greedy policy.
    # We don't choose actions jointly since it need to numerate joint action space with a size of 5^100
    # For each agent choosing action, we assume other agent choose stay there, then find the action with highest reward
    actions = []
    backup_env = copy.deepcopy(env)
    for agent_i in range(env.n_agent):
        rewards_for_each_actions = np.array([-999] * env.act_dims)
        imagine_env = copy.deepcopy(backup_env)
        for action_i in range(env.act_dims):
            dummy_actions = [env.act_dims - 1] * env.n_agent  # means stop
            dummy_actions[agent_i] = action_i
            _, _, rews, _ = imagine_env.step(np.array(dummy_actions))
            imagine_env.from_copy(backup_env)
            reward = rews[agent_i]
            rewards_for_each_actions[action_i] = reward
        best_action_agent_i = rewards_for_each_actions.argmax()
        actions.append(best_action_agent_i)
    return np.array(actions)


def MB_GA_policy(env, max_iter=20):
    # For pop=500, iter=10,
    backup_env = copy.deepcopy(env)
    imagine_env = copy.deepcopy(env)

    def loss_func(actions):
        _, _, rewards, _ = imagine_env.step(actions)
        imagine_env.from_copy(backup_env)
        return - sum(rewards)

    ga = GA(func=loss_func, n_dim=env.n_agent, size_pop=50, max_iter=max_iter, lb=[0] * env.n_agent,
            ub=[env.act_dims - 1] * env.n_agent, precision=1) \
        # .to('cuda:0')
    joint_actions, joint_rewards = ga.run()
    return np.array(joint_actions)
