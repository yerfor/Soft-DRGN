# Scenarios

We use the Scenarios module to implement simulation interaction scenarios. The following is an interface that needs to be implemented in a common scenario.

Note that methods such as get_obs only need to be implemented inside the scene class. Specifically, for Trainer, only Scenario needs to provide the following interfaces

- render(self)： render the current frame of the scenario
- obs, adj = reset(self)：reset the scenario, then obtain the observation and adjacency matrix
- rew, next_obs, next_adj, done = step(self, action)：let the agents exectue the action, then obtain the corresponding reward、next observation

## Data Format Conventions

In the Scenario class, the input action, output reward, obs, adj and other data are all of np.ndarray type, and the dtype is np.float32.

Input shape: for discrete action space, action is [n_agent], for continuous action space, action is [n_agent, action_dim]

Output shape: the size of reward and done is: [n_agent]; the size of obs is [n_agent, obs_dim], if it is a more advanced obs (such as image format), please see utils/replay_buffer.py; the size of adj is [n_agent, n_agent]

```
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
```
