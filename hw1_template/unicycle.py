import numpy as np
import matplotlib.pyplot as plt
import gym


class Unicycle(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super().__init__()
    # The unicycle model has 2 control inputs: linear & angular velocity. We
    # will limit the controller from commanding values beyond these limits
    self.action_space = gym.spaces.Box(
        low=np.array([0, -2*np.pi], dtype=np.float32),
        high=np.array([1., 2*np.pi], dtype=np.float32),
        dtype=np.float32)
    # The unicycle model has 3 states: px, py, theta
    self.observation_space = gym.spaces.Box(
        low=np.array([-100., -100., np.pi], dtype=np.float32),
        high=np.array([100., 100., np.pi], dtype=np.float32),
        dtype=np.float32)

    # We will observe & control the system every self.dt seconds
    self.dt = 0.1

    # The system starts in a random state within these state limits
    self.init_state_range = np.array([
        [-2., 2.],
        [-2., 2.,],
        [-np.pi, np.pi]
    ])

    # The goal position will be within this box, but outside a circle around origin
    self.goal_pos_range = np.array([
        [-10., 10.],
        [-10., 10.,],
    ])
    self.goal_min_radius = 4.


  def step(self, action):
    # Execute one time step within the environment
    new_state = np.empty_like(self.state)
    action = np.clip(action, self.action_space.low, self.action_space.high)

    """Set the values of new_state according to your model here."""
    theta = self.state[2]
    temp = np.array([
        [np.cos(theta), 0.],
        [np.sin(theta), 0.],
        [0., 1.]
    ])

    new_state = np.dot(temp, action)
    while new_state[2] > np.pi:
      new_state[2] -= 2 * np.pi
    while new_state[2] < -np.pi:
      new_state[2] += 2 * np.pi

    self.state = new_state
    self.state_history.append(new_state)

    obs = self.get_obs()
    self.obs_history.append(obs)

    # Check if we have reached the goal
    dist_to_goal = np.linalg.norm(self.goal - new_state[0:2])
    if dist_to_goal < 0.5:
      reward = 1
      done = True
    else:
      reward = 0
      done = False

    info = {}

    return obs, reward, done, info

  def reset(self):
    
    # Set goal that's within the goal limits but outside a circle around origin
    goal = self.get_goal()
    while goal[0]**2+goal[1]**2 < self.goal_min_radius**2:
      goal = self.get_goal()
    self.goal = goal
    
    # Reset the state of the environment to an initial state
    self.state = np.random.uniform(low=self.init_state_range[:, 0],
                                   high=self.init_state_range[:, 1])
    self.state_history = [self.state]
    obs = self.get_obs()
    self.obs_history = [obs]

    return obs
  
  def get_goal(self):
    return np.random.uniform(low=self.goal_pos_range[:, 0], high=self.goal_pos_range[:, 1])

  def get_obs(self):
    return self.state

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    return

  def plot(self):
    plt.figure()
    state_history = np.stack(self.state_history)

    # Draw the path taken by the vehicle
    plt.plot(state_history[:, 0], state_history[:, 1])

    # Draw a green star at the goal position
    plt.plot(self.goal[0], self.goal[1], marker='*', ms=20, c='tab:green')

    # Draw a red x at the starting location
    plt.plot(state_history[0, 0], state_history[0, 1],
             marker='x', ms=20, c='tab:red')
    
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.gca().set_aspect('equal')
    plt.show()
