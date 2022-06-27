import pygame
import numpy as np

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BROWN = (184, 134, 11)
GRAY = (169, 169, 169)
"""
This script implements the renderer of surviving.py, uav_mbs.py, and ctc.py
For renderer code of MAgent scenarios, refer to scenarios/magent_*.py
"""


class GridRenderer:
    def __init__(self, num_grid=40, window_size=(800, 800), grid_width=17, fps=10):
        pygame.init()
        self.num_grid = num_grid
        self.window_size = window_size
        self.grid_width = grid_width
        self.grid_line_width = 2
        self.fps = fps
        self.food_maze = None
        self.resource_maze = None
        self.agent_maze = None

        self.screen = None
        self.background_arr = None

    def init_window(self):
        self.screen = pygame.display.set_mode(self.window_size)
        self.background_arr = self.draw_background()

    def get_mazes(self, food_maze, resource_maze, agent_maze):
        self.food_maze = food_maze
        self.resource_maze = resource_maze
        self.agent_maze = agent_maze
        return food_maze, resource_maze, agent_maze

    def draw_background(self):
        self.screen.fill(WHITE)
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                rect_ij = (self.grid_width * i, self.grid_width * j, self.grid_width, self.grid_width)
                pygame.draw.rect(self.screen, BROWN, rect_ij, width=self.grid_line_width)
        pygame.display.update()
        return pygame.surfarray.array2d(self.screen)

    def draw_foods(self, food_maze, color=GREEN):
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                num_food_on_this_grid = int(food_maze[i][j])
                x_core, y_core = self.grid_width * (i + 0.5), self.grid_width * (j + 0.5)
                food_rect_width = min(num_food_on_this_grid * 3, self.grid_width - self.grid_line_width)
                rect_ij = (
                    x_core - 0.5 * food_rect_width, y_core - 0.5 * food_rect_width, food_rect_width, food_rect_width)
                # rect_ij = (x_core-food_rect_width, y_core-food_rect_width, food_rect_width, food_rect_width)
                if num_food_on_this_grid <= 0:
                    continue
                if color == GREEN:
                    if num_food_on_this_grid == 1:
                        draw_color = (0, 200, 0)
                    elif num_food_on_this_grid == 2:
                        draw_color = (0, 150, 0)
                    elif num_food_on_this_grid >= 3:
                        draw_color = (0, 100, 0)
                else:
                    draw_color = color
                pygame.draw.rect(self.screen, draw_color, rect_ij)

    def draw_resources(self, resource_maze):
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                if resource_maze[i][j] != 0:
                    center_ij = (self.grid_width * (i + 0.5), self.grid_width * (j + 0.5))
                    radius = self.grid_width * 0.5
                    pygame.draw.circle(self.screen, YELLOW, center_ij, radius)

    def draw_agents(self, agent_maze, color=BLACK, raidus_factor=0.2):
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                if agent_maze[i][j] != 0:
                    num_agent_in_this_grid = int(agent_maze[i][j])
                    if num_agent_in_this_grid == 1:
                        center_ij = (self.grid_width * (i + 0.5), self.grid_width * (j + 0.5))
                        radius = self.grid_width * raidus_factor
                        pygame.draw.circle(self.screen, color, center_ij, radius)

    def draw_agents_coverage(self, agent_maze, cover_range=1, color=(255, 0, 0)):
        for i in range(self.num_grid):
            for j in range(self.num_grid):
                if agent_maze[i][j] != 0:
                    rect = (self.grid_width * (i - cover_range), self.grid_width * (j - cover_range),
                            self.grid_width * (cover_range * 2 + 1), self.grid_width * (cover_range * 2 + 1))
                    pygame.draw.rect(self.screen, color, rect, width=3)

    def init(self):
        pygame.display.update()

    def render(self, food_maze, resource_maze, agent_maze, update=True):
        pygame.surfarray.blit_array(self.screen, self.background_arr)
        self.get_mazes(food_maze, resource_maze, agent_maze)  # for future BLIT use
        self.draw_foods(food_maze)
        self.draw_resources(resource_maze)
        self.draw_agents(agent_maze)
        self.draw_agents_coverage(agent_maze, cover_range=1)
        pygame.display.update()
        if update:
            ret_array = pygame.surfarray.array3d(self.screen)
            pygame.time.delay(1000 // self.fps)
            return ret_array

    def render_ctc(self, maze_hunter, maze_red_bank, maze_blue_bank, maze_red_resources, maze_blue_resources,
                   maze_red_treasures, maze_blue_treasures, update=True):
        self.maze_agent_count = np.zeros([self.num_grid, self.num_grid])
        pygame.surfarray.blit_array(self.screen, self.background_arr)
        self.draw_foods(maze_red_treasures, color=RED)
        self.draw_foods(maze_blue_treasures, color=BLUE)
        self.draw_agents(maze_hunter, color=GRAY, raidus_factor=0.3)
        # self.draw_agents_coverage(maze_hunter, cover_range=1)
        self.draw_agents(maze_red_bank, RED, raidus_factor=0.4)
        self.draw_agents(maze_blue_bank, BLUE, raidus_factor=0.4)
        if update:
            pygame.display.update()
            ret_array = pygame.surfarray.array3d(self.screen)
            pygame.time.delay(1000 // self.fps)
            return ret_array

    def render_uav(self, poi_maze, agent_maze, update=True):
        pygame.surfarray.blit_array(self.screen, self.background_arr)
        self.draw_foods(poi_maze)
        self.draw_agents(agent_maze)
        self.draw_agents_coverage(agent_maze, cover_range=1)
        if update:
            pygame.display.update()
            ret_array = pygame.surfarray.array3d(self.screen)
            pygame.time.delay(1000 // self.fps)
            return ret_array

    def render_agent0(self, env, full_connect=True):
        font_str = pygame.font.get_default_font()
        font = pygame.font.Font(font_str, 15)
        agent0_id = 0
        agents = env.ants
        adj = env.get_adj()
        if full_connect:
            adj = np.ones_like(adj)

        neighbors = [i for i, v in enumerate(adj[agent0_id]) if int(v) == 1]
        for neighbor in neighbors:
            if neighbor != 0:
                x, y = agents[neighbor]
                center_ij = (self.grid_width * (x + 0.5), self.grid_width * (y + 0.5))
                radius = self.grid_width * 0.5
                pygame.draw.circle(self.screen, GREEN, center_ij, radius)
                text_surf = font.render(str(neighbor), True, (255, 255, 255))
                text_rect = text_surf.get_rect()
                text_rect.center = center_ij
                self.screen.blit(text_surf, text_rect)
        agent0_x, agent0_y = agents[agent0_id]
        center_ij = (self.grid_width * (agent0_x + 0.5), self.grid_width * (agent0_y + 0.5))
        radius = self.grid_width * 0.5
        pygame.draw.circle(self.screen, RED, center_ij, radius)
        text_surf = font.render(str(agent0_id), True, (255, 255, 255))
        text_rect = text_surf.get_rect()
        text_rect.center = center_ij
        self.screen.blit(text_surf, text_rect)
        pygame.display.update()
        ret_array = pygame.surfarray.array3d(self.screen)
        pygame.time.delay(1000 // self.fps)
        return ret_array
#
# if __name__ == '__main__':
#     from scenarios.ctc import CTC
#     from random import randint
#     env = CTC()
#     render = GridRenderer(num_grid=env.num_grid)
#     mazes = env.get_mazes()
#     while True:
#         render.render_ctc(*mazes)
#         action0 = [randint(0, 6) for _ in range(env.n_hunter)]
#         action1 = [randint(0, 4) for _ in range(env.n_red_bank)]
#         action2 = [randint(0, 4) for _ in range(env.n_blue_bank)]
#         actions = {0: action0, 1: action1, 2: action2}
#         obs, adj, rew, done = env.step(actions)
