class BaseScenario:
    def __init__(self):
        return
        
    def render(self):
        raise NotImplementedError

    def get_obs(self):
        obs = None
        return obs

    def get_adj(self):
        adj = None
        return adj

    def get_done(self):
        done = None
        return done

    def _reset_the_world(self):
        raise NotImplementedError

    def reset(self):
        self._reset_the_world()
        obs = self.get_obs()
        adj = self.get_adj()
        return obs, adj

    def _execute_actions_get_reward(self, actions):
        raise NotImplementedError

    def step(self, actions):
        reward = self._execute_actions_get_reward(actions)
        next_obs = self.get_obs()
        next_adj = self.get_adj()
        next_done = self.get_done()
        return reward, next_obs, next_adj, next_done
