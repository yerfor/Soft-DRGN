import numpy as np

class AgentGroup2D:
    def __init__(self, n_agent, pos=None, vel=None, mass=1):
        self.n_agent = n_agent
        if pos is not None:
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (n_agent, 2)
            self.pos = pos
        else:
            self.pos = np.random.rand(n_agent, 2)
        if vel is not None:
            assert isinstance(vel, np.ndarray)
            assert vel.shape == (n_agent, 2)
        else:
            self.vel = np.zeros([n_agent, 2], dtype=np.float32)
        self.mass = mass

    @property
    def x(self):
        return self.pos[:, 0]

    @property
    def y(self):
        return self.pos[:, 1]

    @property
    def u(self):
        return self.vel[:, 0]

    @property
    def v(self):
        return self.vel[:, 1]
