import rospy
import rospkg
import numpy as np
import yaml
import time
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
    
    # Instantiate the RL interface to the simulation
    rl_interface = RLInterface(
        SimulationInterface(
            MiaHandSetup(hand_config["topics"], hand_config["general"])
        ),
        sim_config
    )
    
    # Instantiate RL env
    rl_env = MiaHandWorldEnv(rl_interface, rl_config, hand_config)  
    #rl_env = MiaHandWorldEnv(hand_config["visual_sensors"], hand_config["limits"], hand_config["general"])  
    
    model = PPO("MultiInputPolicy", rl_env, verbose=1)
    model.learn(total_timesteps=100000)
    
    # r = rl_interface._world_interface._rate
    
    # # Run the episodes
    # for _ in range(rl_config["hyper_params"]["num_episodes"]):
        
    #     # Reset the rl env
    #     obs = rl_env.reset()
        
    #     for _ in range(rl_config["hyper_params"]["max_episode_steps"]):
    #         # Select an action
    #         start_time = time.time()
    #         action = model.predict(obs)
    #         rospy.logwarn_throttle(0.5, "Predict dur: " + str(time.time() - start_time))
    #         # Step the environment
    #         obs, reward, done, info = rl_env.step(action)
            
    #         if rl_interface.step(action[0]) == True:
    #             break
    #         # r.sleep()
        

if __name__ == "__main__":
    rospy.init_node("rl_env", log_level=rospy.INFO)
    np.random.seed(sim_config["seed"])
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass