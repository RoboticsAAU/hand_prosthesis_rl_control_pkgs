import rospy
import rospkg
import numpy as np
import yaml
from stable_baselines3 import PPO
from datetime import datetime

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

with open(rospack.get_path("sim_world") + "/urdf/calibration/joint_limits_wrist.yaml", 'r') as file:
    wrist_limits = yaml.safe_load(file)

with open(rospack.get_path("sim_world") + "/urdf/calibration/joint_limits_fingers.yaml", 'r') as file:
    finger_limits = yaml.safe_load(file)

# with open(rospack.get_path("sim_world") + "/urdf/calibration/joint_limits_cartesian.yaml", 'r') as file:
#     cartesian_limits = yaml.safe_load(file)


# Current date as a string in the format "ddmmyyyy"
algorithm_name = "PPO"
env_name= "mia_hand_rl"
date_string = datetime.now().strftime("%d%m%Y")
log_name = f"{env_name}_{algorithm_name}_{date_string}" 

def main():
    
    # Instantiate the RL interface to the simulation
    rl_interface = RLInterface(
        SimulationInterface(
            MiaHandSetup(hand_config["topics"], hand_config["general"], wrist_limits, finger_limits),
        ),
        sim_config
    )
    
    # Instantiate RL env
    rl_env = MiaHandWorldEnv(rl_interface, rl_config, hand_config, wrist_limits, finger_limits)  

    # Instantiate the PPO model
    model = PPO(
        policy = "MultiInputPolicy",
        env = rl_env,
        batch_size = 64,
        verbose = 1,
        tensorboard_log = rospack.get_path("rl_env") + "/logs",
        device = 'cuda:0'
    )
    
    # Train the model
    model.learn(total_timesteps=100000, tb_log_name=log_name)
        

if __name__ == "__main__":
    rospy.init_node("rl_env", log_level=rospy.INFO)
    np.random.seed(sim_config["seed"])
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass