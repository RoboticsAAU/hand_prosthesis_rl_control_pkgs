import rospy
import rospkg
import numpy as np
import yaml
from stable_baselines3 import PPO


from rl_env.task_envs.mia_hand_task_env import MiaHandWorldEnv
from move_hand.control.move_hand_controller import HandController
from rl_env.setup.hand.mia_hand_setup import MiaHandSetup
from sim_world.world_interfaces.simulation_interface import SimulationInterface
from sim_world.rl_interface.rl_interface import RLInterface

# Load the configuration files
rospack = rospkg.RosPack()
package_path = rospack.get_path("rl_env")
with open(package_path + "/params/rl_params.yaml", 'r') as file:
    rl_config = yaml.safe_load(file)

with open(package_path + "/params/sim_params.yaml", 'r') as file:
    sim_config = yaml.safe_load(file)

with open(package_path + "/params/hand/mia_hand_params.yaml", 'r') as file:
    hand_config = yaml.safe_load(file)

def main():
    # Instantiate RL env
    rl_env = MiaHandWorldEnv(hand_config["visual_sensors"], hand_config["limits"])  
    
    # Instantiate the RL interface to the simulation
    rl_interface = RLInterface(
        SimulationInterface(
            MiaHandSetup(hand_config["topics"], hand_config["general"])
        ),
        rl_env.update,
        sim_config
    )
    
    # Test step of the RL interface
    # input_values = {"action": np.ones(3), 
    #                 "hand_pose": np.concatenate([np.ones(3), np.zeros(4)])}
    # while not rospy.is_shutdown():
    #     rl_interface.step(input_values)
    #     rospy.sleep(0.1)
        
    # Run the episodes
    for _ in range(rl_config["hyper_params"]["num_episodes"]):
        
        rl_interface.update_context()
        
        # Reset the rl env
        rl_env.reset()
        
        
        for _ in range(rl_config["hyper_params"]["max_episode_steps"]):
            # if rl_env.is_done():
            #     break
            
            # TODO: Get the correct action prediction from the RL model
            action = np.zeros(rl_env._action_space.shape)
            
            # Step the environment
            # obs, reward, done, info = rl_env.step(action)

            if rl_interface.step(action) == True:
                break
        
        

if __name__ == "__main__":
    rospy.init_node("rl_env", log_level=rospy.INFO)
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass