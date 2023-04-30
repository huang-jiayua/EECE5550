import numpy as np

class BetterController:
  def __init__(self, controller_name='amazing_controller'):
    """Store any hyperparameters here."""
    self.controller_name = controller_name

  def get_action(self, obs, goal):

    """Your implementation goes here"""
    raise NotImplementedError()
    
    return np.array([linear_speed, angular_speed])