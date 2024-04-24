import rospy
from rospy import Publisher
import rospkg
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from contact_republisher.msg import contacts_msg

class HandSetup(ABC):
    def __init__(self):
        self._name = rospy.get_param('~robot_namespace', None)
        assert self._name is not None, rospy.logerr("Could not get robot namespace")
        rospy.loginfo(self._name)
        
        self._config = {}
        self._hand_rotation = None
        self._contacts_data = contacts_msg()
        self.rospack = rospkg.RosPack()
        
        # Starting ros subscriber to get contacts from contact_republisher_node
        rospy.Subscriber(self._name + "/contact", contacts_msg, self._contact_callback)
    
    @abstractmethod
    def set_action(self, action: np.array) -> None:
        """
        Used to publish states to the hand based on an action (e.g. velocity, force, position)
        """
        pass
    
    @abstractmethod
    def get_subscriber_data(self) -> Dict[str, Any]:
        """
        Get all the subscriber data and return it in a dictionary.
        """
        pass
    
    @abstractmethod
    def _get_subscribers_info(self) -> List[Dict[str, Any]]:
        """
        Get the subscribers information, i.e. topic and message type
        :return: List of the subscribers information as a dict, which includes "topic" and "message_type"
        """
        pass
    
    @abstractmethod
    def _get_publishers(self) -> List[Publisher]:
        """
        Get the publishers
        :return: List of the publishers
        """
        pass
    
    def _get_hand_contact(self) -> contacts_msg:
        
        # Filter contact information to keep only relevant hand-related contacts, i.e. contact between the hand and ground_plane/world.
        # We check if count is equal 1 to avoid collisions between hand and hand itself. 
        hand_contacts = [contact for contact in self._contacts_data.contacts if (contact.collision_1 + contact.collision_2).count(self.name) == 1]

        # Store the filtered contacts in a contacts_msg object
        filt_contacts = contacts_msg()
        filt_contacts.contacts.extend(hand_contacts)
        
        return filt_contacts        
    
    def _check_all_sensors_ready(self) -> None:
        rospy.logdebug("START ALL SENSORS READY")
        
        for subscriber_info in self._get_subscribers_info():
            subscriber_data = None
            while subscriber_data is None and not rospy.is_shutdown():
                try:
                    subscriber_data = rospy.wait_for_message(subscriber_info['topic'], subscriber_info['message_type'], timeout=1.0)
                    rospy.sleep(0.1)
                except:
                    rospy.logerr_throttle_identical(5.0,f"Current {subscriber_info['topic']} not ready yet, retrying for getting {subscriber_info['message_type']}")

        rospy.loginfo("ALL SENSORS READY")
    
    def _wait_for_publishers_connection(self) -> None:
        rospy.logdebug("START ALL PUBLISHERS READY")
        
        rate = rospy.Rate(10)  # 10hz
        for publisher in self._get_publishers():
            while publisher.get_num_connections() == 0 and not rospy.is_shutdown():
                rospy.logdebug(f"No susbribers to {publisher} yet so we wait and try again")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    # This is to avoid error when world is rested, time when backwards.
                    pass
            rospy.logdebug(f"{publisher} Publisher Connected")
        
        rospy.loginfo("ALL PUBLISHERS CONNECTED")

    def _contact_callback(self, data : contacts_msg) -> None:
        self._contacts_data = data

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        rospy.logdebug("Cannot set name directly")
        raise ValueError("Cannot set name directly")
    
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, value):
        rospy.logdebug("Cannot set config directly")
        raise ValueError("Cannot set config directly")
    
    @property
    def hand_rotation(self):
        return self._hand_rotation