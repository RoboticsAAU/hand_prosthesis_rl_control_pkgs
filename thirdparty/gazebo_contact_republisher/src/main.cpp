//
// Created by phil on 24/01/18.
// Modified by ario to include forces on body of contact on 31/05/22

#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/gazebo_config.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Vector3.h>
#include "contact_republisher/contact_msg.h"
#include "contact_republisher/contacts_msg.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
ros::Publisher pub;

std::vector<contact_republisher::contact_msg> contacts_buffer;
size_t buffer_count = 0;
size_t buffer_size = 10;

// Forces callback function
void forcesCb(ConstContactsPtr &_msg)
{
    contact_republisher::contacts_msg contacts_message;
    std::vector<contact_republisher::contact_msg> contacts_list;
    // What to do when callback
    for (int i = 0; i < _msg->contact_size(); ++i)
    {
        contact_republisher::contact_msg contact_message;

        contact_message.collision_1 = _msg->contact(i).collision1();
        contact_message.collision_2 = _msg->contact(i).collision2();

        contact_message.normal[0] = _msg->contact(i).normal().Get(0).x();
        contact_message.normal[1] = _msg->contact(i).normal().Get(0).y();
        contact_message.normal[2] = _msg->contact(i).normal().Get(0).z();

        contact_message.position[0] = _msg->contact(i).position().Get(0).x();
        contact_message.position[1] = _msg->contact(i).position().Get(0).y();
        contact_message.position[2] = _msg->contact(i).position().Get(0).z();

        contact_message.forces_1[0] = _msg->contact(i).wrench(0).body_1_wrench().force().x();
        contact_message.forces_1[1] = _msg->contact(i).wrench(0).body_1_wrench().force().y();
        contact_message.forces_1[2] = _msg->contact(i).wrench(0).body_1_wrench().force().z();

        contact_message.forces_2[0] = _msg->contact(i).wrench(0).body_2_wrench().force().x();
        contact_message.forces_2[1] = _msg->contact(i).wrench(0).body_2_wrench().force().y();
        contact_message.forces_2[2] = _msg->contact(i).wrench(0).body_2_wrench().force().z();

        contact_message.depth = _msg->contact(i).depth().Get(0);

        contacts_list.push_back(contact_message);
    }
    if (_msg->contact_size() == 0)
    {
        contact_republisher::contact_msg contact_message;

        contact_message.collision_1 = "default";
        contact_message.collision_2 = "default";

        contact_message.normal[0] = 0;
        contact_message.normal[1] = 0;
        contact_message.normal[2] = 0;

        contact_message.position[0] = 0;
        contact_message.position[1] = 0;
        contact_message.position[2] = 0;

        contact_message.forces_1[0] = 0;
        contact_message.forces_1[1] = 0;
        contact_message.forces_1[2] = 0;

        contact_message.forces_2[0] = 0;
        contact_message.forces_2[1] = 0;
        contact_message.forces_2[2] = 0;

        contact_message.depth = 0;

        contacts_list.push_back(contact_message);
    }

    contacts_buffer.insert(contacts_buffer.end(), contacts_list.begin(), contacts_list.end());

    if (++buffer_count == buffer_size)
    {
        buffer_count = 0;
        // Processing the contacts
        using count = size_t;
        using index = size_t;
        std::map<std::string, std::pair<index, count>> filtered_contact;
        float max_force = 0.0f;
        ROS_INFO_STREAM("Buffer size: " << contacts_buffer.size());
        // Loop through each contact
        int i = 0;
        for (contact_republisher::contact_msg contact : contacts_buffer)
        {
            std::string unique_key = contact.collision_1;
            unique_key.append(contact.collision_2);
            std::sort(unique_key.begin(), unique_key.end());

            if (filtered_contact.find(unique_key) != filtered_contact.end())
            {
                // If the contact already exists, increment the count
                filtered_contact[unique_key].second++;
            }
            else
            {
                // If the contact does not exist, add it to the map
                filtered_contact[unique_key] = std::make_pair(i, 1);
            }

            // Sum the forces
            contact_republisher::contact_msg best_contact = contacts_buffer[filtered_contact[unique_key].first];
            max_force = std::inner_product(best_contact.forces_1.begin(), best_contact.forces_1.end(), best_contact.forces_1.begin(), 0);
            max_force += std::inner_product(best_contact.forces_2.begin(), best_contact.forces_2.end(), best_contact.forces_2.begin(), 0);
            float force = std::inner_product(contact.forces_1.begin(), contact.forces_1.end(), contact.forces_1.begin(), 0);
            force += std::inner_product(contact.forces_2.begin(), contact.forces_2.end(), contact.forces_2.begin(), 0);

            if (force > max_force)
            {
                filtered_contact[unique_key].first = i;
                max_force = force;
            }
            i++;
        }

        contacts_list.clear();
        ROS_INFO_STREAM("Filtered contacts: " << filtered_contact.size());
        for (const auto &[key, value] : filtered_contact)
        {
            if (static_cast<float>(value.second) > static_cast<float>(buffer_size) / 2.0f)
                contacts_list.push_back(contacts_buffer[value.first]);
        }
        contacts_message.contacts = contacts_list;
        pub.publish(contacts_message);
        contacts_buffer.clear();
    }
}

int main(int _argc, char **_argv)
{

    // Load Gazebo & ROS
    gazebo::client::setup(_argc, _argv);
    ros::init(_argc, _argv, "contact_data");

    // Create Gazebo node and init
    gazebo::transport::NodePtr node(new gazebo::transport::Node());
    node->Init();

    // Create ROS node and init
    ros::NodeHandle n;
    pub = n.advertise<contact_republisher::contacts_msg>("contact", 1);

    // Listen to Gazebo contacts topic
    gazebo::transport::SubscriberPtr sub = node->Subscribe("/gazebo/default/physics/contacts", forcesCb);

    // Busy wait loop...replace with your own code as needed.
    // Busy wait loop...replace with your own code as needed.
    while (true)
    {
        gazebo::common::Time::MSleep(20);

        // Spin ROS (needed for publisher) // (nope its actually for subscribers-calling callbacks ;-) )
        ros::spinOnce();

        // Mayke sure to shut everything down.
    }
    gazebo::client::shutdown();
}
