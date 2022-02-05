# -*- coding: utf-8 -*-
"""
Created on Tue Jun 31 8:34:00 2022

@author: Xiao Peng
"""
import torch
import third_stage.xiao_individualized_model as xim
import third_stage.s3_recieve_cloud as s3rc


HOST = "127.0.0.1"
PATH_data = s3rc.recieve_data(HOST) # recieve the random data
PATH_model = xim.checking(PATH_data)  #
import third_stage.s3_send_cloud as s3sc
s3sc.send_data()   # send the model to the local
