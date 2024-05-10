import rospy
import numpy as np
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.srv import SpawnModel, DeleteModel
from controller_manager_msgs.srv import LoadController
from typing import Union, Dict, List, Type


from move_hand.utils.ros_helper_functions import wait_for_connection
from move_hand.utils.move_helper_functions import next_pose, convert_pose, convert_velocity
from rl_env.setup.hand.hand_setup import HandSetup
from sim_world.world_interfaces.world_interface import WorldInterface
from rl_env.gazebo.gazebo_connection import GazeboConnection
from rl_env.gazebo.controllers_connection import ControllersConnection

class SimulationInterface(WorldInterface):
    def __init__(self, hand_setup : Type[HandSetup]):
        """ hand_name: str is the name of the model in the gazebo world."""
        
        # Initialize the parent class
        super(SimulationInterface, self).__init__(hand_setup)
        
        # Initialise empty set for object names
        self._spawned_obj_names = set()

        # Model state publisher and subscriber
        self._pub_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self._sub_state = rospy.Subscriber('/gazebo/model_states', ModelStates, self._state_callback, queue_size=1)

        # Storing the urdf of the hand
        self._hand_urdf = rospy.get_param("~/robot_description")
        
        # Gazebo connection for pausing/unpausing, configuring physics, etc. 
        self._gazebo_connection = GazeboConnection(False, "NO_RESET_SIM")
        
        # Defining the list of controllers to connect to
        self._controller_list = [publisher.name.split("/")[1] for publisher in self.hand._get_publishers()]
        self._controller_list.append("joint_state_controller") # Since this is not specified in the params file
        rospy.loginfo("Controllers list: " + str(self._controller_list))
        self._controllers_connection = ControllersConnection(namespace=self.hand.name, controllers_list=self._controller_list)
        
        # Set the rate of the controller.
        self.hz = 1000
        self._rate = rospy.Rate(self.hz)  # 1000hz
        
        # Wait for the publishers and subscribers to connect before returning from the constructor. Supply them in a list.
        wait_for_connection([self._pub_state, self._sub_state])
        
        # Wait for the services to spawn and delete objects in the gazebo world
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        self._spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self._spawn_urdf_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self._delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self._load_controller = rospy.ServiceProxy(self.hand.name + '/controller_manager/load_controller', LoadController)
        
        # Initialize variables for saving state information
        self._model_states = ModelStates()
        
        # Hand node for handling spawning
        self._hand_state_pub = rospy.Publisher(self.hand.name + "/poses", ModelStates, queue_size=10)
    
    
    def get_subscriber_data(self) -> List[list]:
        """ Get all the subscriber data and return it in a dictionary. """
        
        # Update the rl_data from mia_hand_setup s.t. it contains object data
        hand_data = self.hand.get_subscriber_data()
        
        # Instantiate subscriber data
        subscriber_data = {"rl_data": {},
                           "move_hand_data": {}}

        # Update rl data with hand and object data
        subscriber_data["rl_data"].update(hand_data)
        subscriber_data["rl_data"].update({"obj_data": self.obj_poses})
        
        # Update move hand (mh) data
        subscriber_data.update({"move_hand_data" : {"pose": self.hand_pose}})
        
        return subscriber_data
    
    def set_pose(self, model_name : str, pose: Union[Pose, np.ndarray], reference_frame: str = 'world'):
        """ Set the pose of the hand to the given position and orientation. Orientation is given in quaternions. """
        try:
            pose = convert_pose(pose)
            self._publish_pose(model_name, pose, reference_frame)
            rospy.sleep(0.15) # Delay is needed, as it takes time to teleport in gazebo
        except Exception as e:
            rospy.logwarn("Failed to set position because: ", e)

    def set_velocity(self, model_name : str, velocity: Union[Twist, np.ndarray]):
        """ Set the velocity of the hand to the given velocity. The velocity can be of type Twist or numpy.ndarray with 6 dimensions."""
        try:
            self._publish_velocity(model_name, velocity)
        except Exception as e:
            rospy.logwarn("Failed to set velocity because: ", e)

    def load_controllers(self, controller_names : List[str]):
        """ Load the controllers for the hand. """
        try:
            
            for controller_name in controller_names:
                self._load_controller(controller_name)
            rospy.loginfo(f"Controller {controller_names} loaded successfully.")
        except Exception as e:
            rospy.logerr("Failed to load controller because: ", e)
            
    def respawn_hand(self, pose : Union[Pose, np.ndarray]):
        self.delete_urdf_model(self.hand.name)
        self.spawn_urdf_model(
            model_name = self.hand.name,
            model_urdf = self._hand_urdf,
            namespace = "/" + self.hand.name + "/",
            pose = pose
        )
        self.load_controllers(self._controller_list)

    def spawn_urdf_model(self, model_name : str, model_urdf: str, namespace : str, pose : Union[Pose, np.ndarray]):
        """ Spawn the object in the gazebo world. """
        # Call the service to spawn the object
        try:
            pose = convert_pose(pose)
            
            self._spawn_urdf_model(model_name, model_urdf, namespace, pose, "world")
            rospy.loginfo(f"Model {model_name} spawned successfully.")
            rospy.sleep(0.15)
        except Exception as e:
            rospy.logerr("Failed to spawn object because: ", e)
    
    def spawn_object(self, model_name : str, model_sdf: str, pose: Union[Pose, np.ndarray]):
        """ Spawn the object in the gazebo world. """
        # Call the service to spawn the object
        try:
            pose = convert_pose(pose)
            
            self._spawn_model(model_name, model_sdf, "", pose, "world")
            self._spawned_obj_names.add(model_name)
            rospy.loginfo(f"Model {model_name} spawned successfully.")
            rospy.sleep(0.15)
        except Exception as e:
            rospy.logerr("Failed to spawn object because: ", e)
        
    def delete_urdf_model(self, model_name: str):
        """ Delete the object from the gazebo world. """
        try:
            self._delete_model(model_name)
            rospy.loginfo(f"Model {model_name} deleted successfully.")
            rospy.sleep(0.15)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to delete object because: ", e)
    
    def delete_object(self, model_name: str):
        """ Delete the object from the gazebo world. """
        try:
            self._delete_model(model_name)
            self._spawned_obj_names.remove(model_name)
            rospy.loginfo(f"Model {model_name} deleted successfully.")
            rospy.sleep(0.15)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to delete object because: ", e)

    def send_hand_poses(self, poses : Union[List[Pose], np.ndarray]):
        num_poses = len(poses) if isinstance(poses, list) else poses.shape[1]
        
        states = ModelStates()
        for i in range(num_poses):
            pose = poses[i] if isinstance(poses, list) else poses[:,i]
            states.name.append(self.hand.name)
            states.pose.append(convert_pose(pose))
            states.twist.append(convert_velocity(np.zeros(6)))
        
        self._hand_state_pub.publish(states)
        
        
    # ------------------- Private methods ------------------- #
    def _publish_velocity(self, model_name : str, velocity: Union[Twist, np.ndarray], ref_frame: str = 'world'):
        """Publish a velocity to the given model in the gazebo world. Includes metadata of the controller in the message."""
        
        velocity = convert_velocity(velocity)
        
        # Initialize a new ModelState instance.
        model_state = ModelState()
        
        # Set the metadata of the message
        model_state.model_name = model_name
        model_state.reference_frame = ref_frame

        # Forward propagate the position of the hand in time and set the next position based on the velocity and delta time.
        model_state.pose = next_pose(velocity, self.hand_pose, 1.0 / float(self.hz))
        
        # Publish the message
        self._pub_state.publish(model_state)

    def _publish_pose(self, model_name : str, pose : Union[Pose, np.ndarray], ref_frame: str = 'world'):
        """ Publish the position of the hand to the gazebo world. Includes metadata of the controller in the message. Overwrites the current position of the hand in the simulation."""            

        pose = convert_pose(pose)

        # Set the data of the message
        model_state = ModelState(model_name=model_name,
                                pose=pose,
                                reference_frame=ref_frame)

        # Publish the message
        self._pub_state.publish(model_state)

    def _state_callback(self, msg : ModelStates):
        """ Callback method for the state subscriber. """
        rospy.logdebug("Received state message")
        self._model_states = msg

    @property
    def hand_pose(self) -> Pose:
        """ Return the current state of the hand. """
        try:
            index = self._model_states.name.index(self.hand.name)
            return self._model_states.pose[index]
        except ValueError:
            rospy.logwarn(f"The gazebo model \"{self.hand.name}\" was not found in the list of list of model states in gazebo")
            return None

    @property
    def obj_poses(self) -> Dict[str, Pose]:
        """ Return poses of objects in the environment. """
        try:
            obj_tuples = [(obj_name, self._model_states.name.index(obj_name)) for obj_name in self._spawned_obj_names]
            return {name : self._model_states.pose[index] for name, index in obj_tuples}
        except ValueError as e:
            rospy.logwarn(f"An index exception has occurred when indexing objects: {e}")
            return None


if __name__ == '__main__':
    print("{} should not be run as a script...".format(__file__))