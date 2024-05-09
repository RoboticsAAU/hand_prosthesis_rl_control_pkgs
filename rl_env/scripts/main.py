import rospy
import rospkg
import numpy as np
import yaml
import torch as th
from stable_baselines3.ppo import PPO
from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime

from rl_env.task_envs.mia_hand_task_env import MiaHandWorldEnv
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

with open(rospack.get_path("sim_world") + "/config/joint_limits.yaml", 'r') as file:
    joint_limits = yaml.safe_load(file)


# Continue Learning with the this model: If no model should be loaded, set model_to_load to None
model_to_load = 'rl_model_09052024_8500_steps.zip'
# model_to_load = None
log_to_load = 'mia_hand_rl_PPO_09052024_1'


checkpoint_location = package_path + "/logging/checkpoints"
tensorboard_log = package_path + "/logging/tb_events/"
tb_log_to_load = tensorboard_log + log_to_load

# Current date as a string in the format "ddmmyyyy"
algorithm_name = "PPO"
env_name= "mia_hand_rl"
datetime_string = datetime.now().strftime("%Y%m%D_%H%M%S")
tb_log_name = package_path + "/logging/tb_events/" + f"{env_name}_{algorithm_name}_{datetime_string}" 

steps_per_episode = sim_config["move_hand"]["num_points"]/sim_config["move_hand"]["traj_buffer_size"]

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq = 10*steps_per_episode,
    name_prefix = "rl_model" + f"_{datetime_string}",
    save_path = checkpoint_location,
    save_replay_buffer = False,
    save_vecnormalize = False,
)

def get_3d_policy_kwargs(extractor_name) -> dict:
    feature_extractor_class = PointNetImaginationExtractorGP
    feature_extractor_kwargs = {"pc_key": "camera-point_cloud",
                                "extractor_name": extractor_name,
                                "imagination_keys": ["imagined"],
                                "state_key": "state"}

    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "activation_fn": th.nn.ReLU,
    }
    return policy_kwargs


def main():
    
    # Instantiate the RL interface to the simulation
    rl_interface = RLInterface(
        SimulationInterface(
            MiaHandSetup(hand_config, joint_limits),
        ),
        sim_config
    )
    
    # Instantiate RL env
    rl_env = MiaHandWorldEnv(rl_interface, rl_config)

    # setting device on GPU if available, else CPU
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
        
    # Instantiate the PPO model
    if model_to_load is not None:
        model = PPO.load(
            path=checkpoint_location + "/" + model_to_load,
            tensorboard_log=tb_log_to_load
            )
        model.set_env(env=rl_env)
    else:
        model = PPO(
            policy = "MultiInputPolicy",
            tensorboard_log=tensorboard_log,
            env = rl_env,
            verbose = 1, 
            device = device,
            policy_kwargs=get_3d_policy_kwargs(extractor_name="smallpn"), # Can either be "smallpn", "mediumpn" or "largepn". See sb3.common.torch_layers.py 
            **rl_config["hyper_params"]
        )
    
    # Train the model
    timesteps = rl_config["general"]["num_episodes"]*steps_per_episode
    model.learn(total_timesteps=timesteps, tb_log_name=tb_log_name, callback=checkpoint_callback)
    

if __name__ == "__main__":
    rospy.init_node("rl_env", log_level=rospy.INFO)
    np.random.seed(sim_config["seed"])
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass