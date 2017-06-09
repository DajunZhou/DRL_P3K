#!/usr/bin/env python

import sys
import rospy
import os
import numpy as np
from configure import configure

# os.system('roslaunch P3K_world P3K_world.launch')
# os.system('rosrun gazebo_ros spawn_model -x 2 -y 2 -z 1 -file `rospack find P3K_description`/urdf/myp3at_kinect1.urdf -urdf -model piokinect2 -robot_namespace piokinect2')

number_of_food = configure.number_of_food
number_of_bots = configure.number_of_bots
number_of_pred = configure.number_of_pred
arena_width = configure.arena_width

def spawn_model(type_spawn, pos, i):
    rgsp = 'rosrun gazebo_ros spawn_model -x ' +str(pos[0])+ ' -y ' +str(pos[1])+ ' -z ' +str(pos[2]) + ' -Y ' + str(pos[3])
    rgsp_pSdf = rgsp + ' -file `rospack find P3K_description`/urdf/'
    look_up = {'food':rgsp_pSdf + 'food.sdf' + ' -sdf -model food' + str(i) + ' -robot_namespace food' + str(i),
               'piokinect' :rgsp_pSdf + 'myp3at_kinect1.urdf' + ' -urdf -model piokinect' + str(i) + ' -robot_namespace piokinect' + str(i),
               'predator' :rgsp_pSdf + 'p3k_predator.sdf' + ' -sdf -model predator' + str(i) + ' -robot_namespace predator' + str(i)}
    spawn_text = look_up[type_spawn]
    os.system(spawn_text)
    return spawn_text

def spawn_all(type_spawn, num):
    spawn_texts = {}
    # init_i = {'food':0, 'piokinect':0, 'predator':0}
    if not num:
        return spawn_texts
    for i in xrange(0,num):
        position = np.random.uniform([-arena_width,-arena_width,0.1, -3.1416],[arena_width,arena_width,0.2,3.1416],size=4)
        spawn_texts[type_spawn + str(i)] = spawn_model(type_spawn, position, i)

    return spawn_texts


def main(args):

    result_food = spawn_all("food", number_of_food)
    result_bots = spawn_all("piokinect", number_of_bots)
    result_pred = spawn_all("predator", number_of_pred)


if __name__ == '__main__':
    main(sys.argv)