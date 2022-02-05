# -*- coding: utf-8 -*-
"""
Created on Wed Feb 2 11:41:00 2022

@author: Lin Ze

This file exists on the local server and is used to receive models 
from other robotic arms in the same scene, and fuse its models with 
its own model to form a new model.

This algorithm can be used to learn ahead of time from locally unseen errors.
"""

import  first_stage.xiao_fault_merge as xfm
import first_stage.s1_recieve as sr
HOST = "127.0.0.1"  # Enter the IP address of the object you want to learn
PATH = sr.recieve_model(HOST) # Extract the obtained model path
xfm.merge_model(PATH) # Fusion models and testing

