import rospy
import rospkg
import numpy as np
import yaml
import torch as th
import glob
import os
from datetime import datetime
from pathlib import Path
from stable_baselines3.ppo import PPO
from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_env.task_envs.mia_hand_task_env import MiaHandWorldEnv
from rl_env.setup.hand.mia_hand_setup import MiaHandSetup
from sim_world.world_interfaces.simulation_interface import SimulationInterface
from sim_world.rl_interface.rl_interface import RLInterface


# Load the configuration files
rospack = rospkg.RosPack()
package_path = Path(rospack.get_path("rl_env"))
with open(package_path.joinpath("params/rl_params.yaml"), 'r') as file:
    rl_config = yaml.safe_load(file)

with open(package_path.joinpath("params/sim_params.yaml"), 'r') as file:
    sim_config = yaml.safe_load(file)

with open(package_path.joinpath("params/hand/mia_hand_params.yaml"), 'r') as file:
    hand_config = yaml.safe_load(file)

with open(Path(rospack.get_path("sim_world")).joinpath("config/joint_limits.yaml"), 'r') as file:
    joint_limits = yaml.safe_load(file)

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

train = True

# Continue learning with this model: If no model should be loaded, set model_to_load to None
# model_to_load = None
# log_to_load = None
# model_to_load = "mia_hand_rl_PPO_20240513_170245_6500_steps.zip"
# log_to_load = "mia_hand_rl_PPO_20240513_170245_0/events.out.tfevents.1715619799.rog-laptop.41667.0"

checkpoint_dir = package_path.joinpath("logging/checkpoints")
tensorboard_dir = package_path.joinpath("logging/tb_events")

env_name= "mia_hand_rl_PPO"

# Check if folder is empty
checkpoints = glob.glob(str(checkpoint_dir.joinpath("*.zip")))
checkpoints.sort(key=lambda x: os.path.getmtime(x))

if len(checkpoints) == 0:
    model_to_load = None
    log_to_load = None
    initial_steps = 0
    # Current date as a string in the format "ddmmyyyy"
    datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_identifier = f"{env_name}_{datetime_string}"
    
else:
    model_to_load = Path(checkpoints[-1]).name
    model_identifier = '_'.join(model_to_load.split("_")[:-2])
    initial_steps = int(model_to_load.split("_")[-2])
    log_to_load = glob.glob(str(tensorboard_dir.joinpath(f"{model_identifier}*/*")), recursive=True)[0]
    log_to_load = '/'.join(log_to_load.split("/")[-2:])


steps_per_episode = sim_config["move_hand"]["num_points"] / sim_config["move_hand"]["traj_buffer_size"]
initial_episode = initial_steps//steps_per_episode

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq = 10*steps_per_episode,
    name_prefix = model_identifier,
    save_path = checkpoint_dir,
    save_replay_buffer = False,
    save_vecnormalize = False,
)


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
    rl_env.set_episode_count(initial_episode)
    # while True:

    # setting device on GPU if available, else CPU
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    timesteps = rl_config["general"]["num_episodes"]*steps_per_episode
    
    reset_num_timesteps = None    
    
    # Instantiate the PPO model
    if model_to_load is not None:
        model = PPO.load(
            path = checkpoint_dir.joinpath(model_to_load),
            tensorboard_log = tensorboard_dir.joinpath(log_to_load),
            verbose = 1, 
            device = device,
            policy_kwargs = get_3d_policy_kwargs(extractor_name="smallpn"), # Can either be "smallpn", "mediumpn" or "largepn". See sb3.common.torch_layers.py 
            **rl_config["hyper_params"]
            )
        model.set_env(env=rl_env)
        
        reset_num_timesteps = False
        
    else:
        model = PPO(
            policy = "MultiInputPolicy",
            tensorboard_log = tensorboard_dir,
            env = rl_env,
            verbose = 1, 
            device = device,
            policy_kwargs = get_3d_policy_kwargs(extractor_name="smallpn"), # Can either be "smallpn", "mediumpn" or "largepn". See sb3.common.torch_layers.py 
            **rl_config["hyper_params"]
        )

        reset_num_timesteps = True
    
    if train is True:
        # Train the model
        model.learn(
            total_timesteps = timesteps,
            tb_log_name = tensorboard_dir.joinpath(model_identifier),
            reset_num_timesteps = reset_num_timesteps,
            callback = checkpoint_callback
        )
        
    else:
        for episode in range(rl_config["general"]["num_episodes"]):
            obs, info = rl_env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = rl_env.step(action)
                rospy.loginfo(f"Episode: {episode}, Reward: {reward}")
    

if __name__ == "__main__":
    # rospy.init_node("rl_env", log_level=rospy.INFO)
    np.random.seed(sim_config["seed"])
    
    try:
        # main()
        pass
    except rospy.ROSInterruptException:
        pass