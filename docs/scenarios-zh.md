# Scenarios模块

我们使用Scenarios模块实现仿真交互场景，下面是一个常见的场景需要实现的接口。

注意get_obs等方法只要在场景类内部实现就好，具体来说，对Trainer来说，只需要Scenario提供以下接口：

- render(self)： 渲染环境当前帧
- obs, adj = reset(self)：重置环境，获取observation和adjacency matrix
- rew, next_obs, next_adj, done = step(self, action)：让智能体执行指定的action，获得对应的reward、next observation等信息

## 数据格式约定

Scenario类，输入的action、输出的reward、obs、adj等数据，都是np.ndarray类型，dtype为np.float32。

输入尺寸：对离散动作空间，action为[n_agent]，对连续动作空间，action为[n_agent, action_dim]

输出尺寸：reward、done的尺寸为：[n_agent]；obs的尺寸为[n_agent, obs_dim]，如果是更进阶的obs（如图像格式的），请看utils/replay_buffer.py；adj的尺寸为[n_agent, n_agent]。

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

