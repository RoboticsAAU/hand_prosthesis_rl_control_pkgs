from sim_world.world_interfaces.world_interface import WorldInterface
from typing import Type, Dict, Callable

class RLInterface():
    def __init__(self, update_methods: Dict[str, Callable], world_interface: Type[WorldInterface]):
        self.update_methods = update_methods
        self.world_interface = world_interface

    def update(self):
        # Extract all the values from the interface and put them in a dictionary
        # Some values may be set to none depending on the interface, need to make sure the update methods can handle this using checks. 
        values = self.world_interface.update()
        for method in self.update_methods.values():
            method(values)