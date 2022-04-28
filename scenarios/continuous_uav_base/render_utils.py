import pygame
import numpy as np
import cv2

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BROWN = (184, 134, 11)
GRAY = (169, 169, 169)
DISPLAY_BATTERY = True


class ContinuousWorldRenderer:
    def __init__(self, num_grid=500, obs_range=25, window_size=(900, 800), scale_factor=1, fps=20):
        pygame.init()
        self.num_grid = num_grid
        self.obs_range = obs_range
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.fps = fps
        self.food_maze = None
        self.resource_maze = None
        self.agent_maze = None

        self.screen = pygame.display.set_mode(self.window_size)
        self.background_arr = self.draw_background()
        self.init()

    def get_mazes(self, food_maze, resource_maze, agent_maze):
        self.food_maze = food_maze
        self.resource_maze = resource_maze
        self.agent_maze = agent_maze
        return food_maze, resource_maze, agent_maze

    def draw_background(self):
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, BLACK,
                         [0, 0, self.scale_factor * (self.num_grid + 2 * self.obs_range),
                          self.scale_factor * self.obs_range])
        pygame.draw.rect(self.screen, BLACK, [0, 0, self.obs_range * self.scale_factor,
                                              self.scale_factor * (self.num_grid + 2 * self.obs_range)])
        pygame.draw.rect(self.screen, BLACK,
                         [self.scale_factor * (self.num_grid + self.obs_range), 0, self.obs_range * self.scale_factor,
                          self.scale_factor * (self.num_grid + 2 * self.obs_range)])
        pygame.draw.rect(self.screen, BLACK,
                         [0, self.scale_factor * (self.num_grid + self.obs_range),
                          self.scale_factor * (self.num_grid + 2 * self.obs_range), self.obs_range * self.scale_factor])
        pygame.display.update()
        return pygame.surfarray.array2d(self.screen)

    def draw_foods(self, food_pos, color=GREEN, raidius=1.5):
        for p in food_pos:
            pygame.draw.circle(self.screen, color, (p * self.num_grid + self.obs_range)*self.scale_factor, raidius*self.scale_factor)

    def draw_charging_stations(self, charger_pos, color=BLUE, raidius=3):
        for p in charger_pos:
            pygame.draw.circle(self.screen, color, (p * self.num_grid + self.obs_range)*self.scale_factor, raidius*self.scale_factor)

    def draw_battery_usage(self, battery_usage, font, color=BLACK):
        for i in range(len(battery_usage)):
            color = RED if battery_usage[i] <= 0 else BLACK
            text_surf = font.render("UAV {:d}: {:d}".format((i), battery_usage[i]), True, color)
            text_rect = text_surf.get_rect()
            text_rect.center = (750, 200+25*i)
            self.screen.blit(text_surf, text_rect)

    def draw_agents(self, agent_pos, color=BLUE, radius=2, agent_surface=None, show_id=True):
        for id,p in enumerate(agent_pos):
            if agent_surface:
                core_pos = (p * self.num_grid + self.obs_range) * self.scale_factor
                r = self.scale_factor*radius
                self.screen.blit(agent_surface,[core_pos[0]-r, core_pos[1]-r,2*r,2*r])
            else:
                pygame.draw.circle(self.screen, color, (p * self.num_grid + self.obs_range) * self.scale_factor, radius * self.scale_factor)
            if show_id:
                font_str = pygame.font.get_default_font()
                font = pygame.font.Font(font_str, 15)
                text_surf = font.render(str(id), True, BLACK)
                text_rect = text_surf.get_rect()
                text_rect.center = (core_pos[0]+10, core_pos[1]+10)
                self.screen.blit(text_surf, text_rect)

    def draw_agents_coverage(self, agent_pos, cover_range=1, color=BLUE):
        for p in agent_pos:
            pygame.draw.circle(self.screen, color, (p * self.num_grid + self.obs_range)*self.scale_factor, cover_range*self.scale_factor, width=1)

    def init(self):
        pygame.display.update()

    def render_uav(self, render_info, update=True,agent_surface=None):

        poi_pos = render_info['poi_pos']
        agent_pos = render_info['agent_pos']
        rect_obstacles = render_info['rect_obstacles']
        circle_obstacles = render_info['circle_obstacles']
        poi_cover_id = render_info['poi_cover_id']
        poi_cover_percent = render_info['poi_cover_percent']
        episode_coverage_item = render_info['episode_coverage_item']
        episode_fairness_item = render_info['episode_fairness_item']
        energy_consumption = render_info['energy_consumption']
        timeslot = render_info['timeslot']
        pygame.surfarray.blit_array(self.screen, self.background_arr)
        covered_poi_pos = []
        uncovered_poi_pos = []
        for i, pos in enumerate(poi_pos):
            if poi_cover_id[i] != 0:
                covered_poi_pos.append(pos)
            else:
                uncovered_poi_pos.append(pos)
        self.draw_foods(covered_poi_pos, color=GREEN)
        self.draw_foods(uncovered_poi_pos, color=GRAY)

        for (x_min,y_min,height,width) in rect_obstacles:
            rect = np.array([x_min, y_min, height, width])*self.scale_factor
            pygame.draw.rect(self.screen, BLACK, rect)
        for (x,y,r) in circle_obstacles:
            pygame.draw.circle(self.screen,BLACK, np.array([x,y])*self.scale_factor, r*self.scale_factor)

        self.draw_agents(agent_pos, agent_surface=agent_surface)
        self.draw_agents_coverage(agent_pos, cover_range=18, color=BROWN)  # COMM
        self.draw_agents_coverage(agent_pos, cover_range=13, color=BLUE)  # OBS
        self.draw_agents_coverage(agent_pos, cover_range=10, color=RED)  # COVER

        font_str = pygame.font.get_default_font()
        font = pygame.font.Font(font_str, 15)
        text_surf = font.render("Timeslot: {}".format(timeslot), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (680,25)
        self.screen.blit(text_surf, text_rect)

        font_str = pygame.font.get_default_font()
        font = pygame.font.Font(font_str, 15)
        text_surf = font.render("PoI Covered Percentage: {:.2f}".format(poi_cover_percent), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (680,50)
        self.screen.blit(text_surf, text_rect)

        text_surf = font.render("Final Coverage Index: {:.2f}".format(episode_coverage_item), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (680,100)
        self.screen.blit(text_surf, text_rect)

        text_surf = font.render("Final Fairness Index: {:.2f}".format(episode_fairness_item), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (680,150)
        self.screen.blit(text_surf, text_rect)

        text_surf = font.render("Current Energy Index: {:.2f}".format(energy_consumption), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (680,200)
        self.screen.blit(text_surf, text_rect)

        if update:
            pygame.display.update()
            ret_array = pygame.surfarray.array3d(self.screen)
            pygame.time.delay(1000 // self.fps)
            return ret_array

    def render_mcs(self, render_info, update=True,agent_surface=None):

        poi_pos = render_info['poi_pos']
        agent_pos = render_info['agent_pos']
        rect_obstacles = render_info['rect_obstacles']
        circle_obstacles = render_info['circle_obstacles']
        poi_cover_id = render_info['poi_cover_id']
        poi_cover_percent = render_info['poi_cover_percent']
        episode_coverage_item = render_info['episode_coverage_item']
        episode_fairness_item = render_info['episode_fairness_item']
        energy_consumption = render_info['energy_consumption']
        timeslot = render_info['timeslot']
        charger_pos = render_info['charger_pos']
        battery_usage = render_info['battery']         # UAV如何显示电量

        pygame.surfarray.blit_array(self.screen, self.background_arr)
        covered_poi_pos = []
        uncovered_poi_pos = []
        for i, pos in enumerate(poi_pos):
            if poi_cover_id[i] != 0:
                covered_poi_pos.append(pos)
            else:
                uncovered_poi_pos.append(pos)
        self.draw_foods(covered_poi_pos, color=GREEN)
        self.draw_foods(uncovered_poi_pos, color=GRAY)
        self.draw_charging_stations(charger_pos, color=BLUE)

        for (x_min,y_min,height,width) in rect_obstacles:
            rect = np.array([x_min, y_min, height, width])*self.scale_factor
            pygame.draw.rect(self.screen, BLACK, rect)
        for (x,y,r) in circle_obstacles:
            pygame.draw.circle(self.screen,BLACK, np.array([x,y])*self.scale_factor, r*self.scale_factor)

        self.draw_agents(agent_pos, agent_surface=agent_surface)
        self.draw_agents_coverage(agent_pos, cover_range=18, color=BROWN)  # COMM
        self.draw_agents_coverage(agent_pos, cover_range=13, color=BLUE)  # OBS
        self.draw_agents_coverage(agent_pos, cover_range=10, color=RED)  # COVER
        # self.draw_agents_coverage(charger_pos, cover_range=5,color=RED)  # Draw Charging Range

        font_str = pygame.font.get_default_font()
        font = pygame.font.Font(font_str, 15)
        text_surf = font.render("Timeslot: {}".format(timeslot), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (790, 25)
        self.screen.blit(text_surf, text_rect)

        font_str = pygame.font.get_default_font()
        font = pygame.font.Font(font_str, 15)
        text_surf = font.render("PoI Covered Percentage: {:.2f}".format(poi_cover_percent), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (790, 50)
        self.screen.blit(text_surf, text_rect)

        text_surf = font.render("Final Coverage Index: {:.2f}".format(episode_coverage_item), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (790, 75)
        self.screen.blit(text_surf, text_rect)

        text_surf = font.render("Final Fairness Index: {:.2f}".format(episode_fairness_item), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (790, 100)
        self.screen.blit(text_surf, text_rect)

        text_surf = font.render("Current Energy Index: {:.2f}".format(energy_consumption), True, BLACK)
        text_rect = text_surf.get_rect()
        text_rect.center = (790, 125)
        self.screen.blit(text_surf, text_rect)

        if DISPLAY_BATTERY:
            self.draw_battery_usage(battery_usage, font, color=BLACK)

        if update:
            pygame.display.update()
            ret_array = pygame.surfarray.array3d(self.screen)
            pygame.time.delay(1000 // self.fps)
            return ret_array