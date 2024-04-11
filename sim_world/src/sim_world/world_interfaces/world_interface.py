from rl_env.setup.hand.hand_setup import HandSetup
from abc import ABC, abstractmethod
from typing import Type

class WorldInterface():
    def __init__(self, hand_setup: Type[HandSetup]):
        self.hand = hand_setup
        self.check_system_ready()
        
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_subscriber_data(self):
        pass
    
    def check_system_ready(self):
        self.hand._check_all_sensors_ready()
        self.hand._wait_for_publishers_connection()
