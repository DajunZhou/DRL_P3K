#!/usr/bin/env python

import rospy
# from std_msgs.msg import String
# from gazebo_msgs.msg import ContactsState
# from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates
from p3k import piokinect
from configure import configure
from GA3C_P3K.srv import *
from food import food

# import re
rospy.init_node('rewards', anonymous=True)
initialised = False
velocity_updated = False
sensor_range = 2
mode = configure.mode
number_of_food = configure.number_of_food
number_of_bots = configure.number_of_bots
number_of_pred = configure.number_of_pred
arena_width = configure.arena_width
# SWARMBOT_GAZEBO_init_i = {'food':0, 'piokinect':0, 'predator':0}

def create_models(type_model, num):
    res = {}
    if num < 1 :
        return res
    if type_model == 'food':
        for i in xrange(0, num):
            res[i] = food(i, 0,0,1, 1)
            rospy.sleep(0.05)
            res[i].random_relocate(arena_width)
            rospy.sleep(0.05)
    elif type_model == 'predator':
        for i in xrange(0, num):
            res[i] = piokinect(i, 0,0,1, -20, 'predator')
            rospy.sleep(0.05)
            res[i].random_relocate(arena_width)
    elif type_model == 'piokinect':
        for i in xrange(0, num):
            global foods
            res[i] = piokinect(i, 0,0,1, -5, 'piokinect')
            rospy.sleep(0.05)
            res[i].random_relocate(arena_width)
    return res

foods = create_models("food", number_of_food)
predators = create_models("predator", number_of_pred)
piokinects = create_models("piokinect", number_of_bots)

def have_consume_food_handler(request):
    global foods
    foods[request.id].have_consumed()
    return Data_requestResponse(True, None)

server_consume_food = rospy.Service('/Have_Consume_Food', Data_request, have_consume_food_handler)

def food_relocate_request_handler(request):
    global foods
    foods[request.id].random_relocate(arena_width)
    return Data_requestResponse(True, None)

server_food_relocate = rospy.Service('/food_relocate_request', Data_request, food_relocate_request_handler)

def random_relocate_service_handler(request):
    global piokinects
    piokinects[request.id].random_relocate(arena_width)
    return Data_requestResponse(True, None)

server_relocate = rospy.Service('/random_relocate_request', Data_request, random_relocate_service_handler)

def speed_request_service_handler(request):
    # global piokinects
    # response.data = piokinects[request.id].speed
    return Data_requestResponse(True, piokinects[request.id].speed)

server_speed = rospy.Service('/speed_request', Data_request, speed_request_service_handler)

def calculate_energy(id):
    r = rospy.Rate(100)
    global piokinects
    req_time = 0
    while (not piokinects[id].collision_updated) and (req_time < 300):
        r.sleep()
        req_time += 1
        # print "wait energy"
    piokinects[id].collision_updated = False

def energy_service_handler(request):
    calculate_energy(request.id)
    # response.data = piokinects[request.id].energy
    return Data_requestResponse(True, piokinects[request.id].energy)

server_energy = rospy.Service('/energy_request', Data_request, energy_service_handler)

model_states_initialised = False

def table_invert(t):
    s = {}
    for k, v in enumerate(t):
        s[v] = k
    return s

def SGDQN_add_model_id(res_i, type_model, num, index_lookup):
    for i in xrange(0, num):
        res_i[i].model_id = index_lookup[res_i[i].model_name]
    return res_i

def SGDQN_add_infos_msgs(res, type_model, num, msg):
    for i in xrange(0, num):
        if type_model == 'food':
            res[i].position[0] = msg.pose[res[i].model_id].position.x
            res[i].position[1] = msg.pose[res[i].model_id].position.y
            res[i].position[2] = msg.pose[res[i].model_id].position.z
            res[i].position[3] = msg.pose[res[i].model_id].orientation.z
        elif type_model == 'piokinect' or type_model == 'predator':
            res[i].velocity[0] = msg.twist[res[i].model_id].linear.x
            res[i].velocity[1] = msg.twist[res[i].model_id].linear.y
            res[i].velocity[2] = msg.twist[res[i].model_id].linear.z
            res[i].position[0] = msg.pose[res[i].model_id].position.x
            res[i].position[1] = msg.pose[res[i].model_id].position.y
            res[i].position[2] = msg.pose[res[i].model_id].position.z
            res[i].position[3] = msg.pose[res[i].model_id].orientation.z
    return res

def states_callback(msg):
    # print "I get!"
    global model_states_initialised, velocity_updated
    global foods, predators, piokinects
    if not model_states_initialised:
        index_lookup = table_invert(msg.name)
        foods = SGDQN_add_model_id(foods, "food", number_of_food, index_lookup)
        predators = SGDQN_add_model_id(predators, "predator", number_of_pred, index_lookup)
        piokinects = SGDQN_add_model_id(piokinects, "piokinect", number_of_bots, index_lookup)

    model_states_initialised = True

    foods = SGDQN_add_infos_msgs(foods, "food", number_of_food, msg)
    predators = SGDQN_add_infos_msgs(predators, "predator", number_of_pred, msg)
    piokinects = SGDQN_add_infos_msgs(piokinects, "piokinect", number_of_bots, msg)

    velocity_updated = True

model_state_subscriber = rospy.Subscriber('/throttled_model_states', ModelStates, states_callback, queue_size=100)

Rate = rospy.Rate(100)
while not rospy.is_shutdown():
    Rate.sleep()

#
# if mode == 0:
#     while not rospy.is_shutdown():
#         Rate.sleep()
#         for i in xrange(0, number_of_food):
#             if not foods[i].edible:
#                 foods[i].edible = os.clock() - foods[i].immunity_start_time > foods[i].immunity_duration
# elif mode ==1:
#     training_range = sensor_range + 1
#     while not rospy.is_shutdown():
#         Rate.sleep()
#         for i in xrange(0, number_of_food):
#             if not foods[i].edible:
#                 foods[i].edible = os.clock() - foods[i].immunity_start_time > foods[i].immunity_duration
#
#         for j in xrange(0, number_of_food):
#             pass

server_energy.shutdown()
server_relocate.shutdown()
server_speed.shutdown()