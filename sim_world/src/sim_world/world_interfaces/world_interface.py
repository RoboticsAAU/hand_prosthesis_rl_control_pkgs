from rl_env.setup.hand.hand_setup import HandSetup
from abc import ABC, abstractmethod

class WorldInterface():
    def __init__(self, hand_setup: HandSetup):
        self.hand = hand_setup
    
    @abstractmethod
    def reset(self):
        pass
    
    def check_system_ready(self):
        self.hand._check_all_sensors_ready()
        self.hand._wait_for_publishers_connection()
