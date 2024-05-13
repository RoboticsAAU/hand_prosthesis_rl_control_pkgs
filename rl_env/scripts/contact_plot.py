#!/usr/bin/env python3

import rospy
import numpy as np
import rospkg
import glob
from pathlib import Path
from contact_republisher.msg import contacts_msg
import matplotlib.pyplot as plt
from time import time

def main():
    
    _contacts_data = []
    timestamp = time()
    
    def _contact_callback(data : contacts_msg) -> None:
        nonlocal timestamp, _contacts_data
        val = 0
        count = 0
        for contact in data.contacts:
            # Early continue if the contact is with the ground plane or if force vector is zero (spurious contact)
            if ("ground_plane" in (contact.collision_1 + contact.collision_2)):
                continue
            
            if (contact.collision_1 + contact.collision_2).count("thumb_fle") > 0:
                val = 1
        
        _contacts_data.append(val)
                
        if time() - timestamp >= 1:
            print("Saving figure...")
            plt.scatter(range(len(_contacts_data)), _contacts_data, marker='.', s=0.1)
            plt.savefig("contact_plot.png")
            plt.close()
            print("Contacts length: ", len(_contacts_data)) 
            print("Contacts percentage: ", _contacts_data.count(1) / len(_contacts_data) * 100.0, "%")
            timestamp = time()
            _contacts_data.clear()
    
    rospy.Subscriber("mia_hand_sim" + "/contact", contacts_msg, _contact_callback)
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("mia_hand_rl_env")
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass