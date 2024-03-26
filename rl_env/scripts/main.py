from task_envs.mia_hand_task_env import MiaHandWorldEnv
from move_hand.move_hand_controller import HandController
from move_hand.gazebo_interface import GazeboInterface
from stablebaselines3 import PPO
from config.config.hand import MiaHandConfig
from types import SimpleNamespace
import yaml

with open('hand_params.yaml', 'r') as file:
    hand_params = yaml.safe_load(file)

with open('rl_params.yaml', 'r') as file:
    rl_params = yaml.safe_load(file)

with open('sim_params.yaml', 'r') as file:
    sim_params = yaml.safe_load(file)

# Meta information used for the RL training
rl_props = SimpleNamespace(
    num_episodes = 100,
    log_dir = "logs/",
    ckpt_dir = "checkpoints/",
    max_episodes = 10000
)


def main():

    # Instantiate gazebo interface
    gazebo_interface = GazeboInterface(MiaHandConfig(hand_params))
    
    # Instantiate RL env
    rl_config = {"config_camera": None,
                 "config_imagined": None}
    rl_env = MiaHandWorldEnv(rl_config)    
    
    # Instantiate the hand controller
    hand_controller = HandController()
    
    # TODO: Instantiate the graspit controller
    
    

    # Run the episodes
    while rl_props.episodes < rl_props.max_episodes:
        # Reset the env
        rl_env.reset()
        # Step the environment
        rl_env.step()
        # Update the RL model



if __name__ == "__main__":
    main()