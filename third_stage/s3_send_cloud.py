# _*_ coding:utf-8 _*_
"""
Created on Wed Feb 2 11:41:00 2022

@author: Lin Ze

This file is used to send models.
"""
from socket import *
import _thread
def send_data():
    HOST = "127.0.0.1"
    PORT = 23333
    ADDR = (HOST, PORT)
    client = socket(AF_INET, SOCK_STREAM)
    client.connect(ADDR)
    print('Start sending')
    with open('../global_models/net_xiao_global_indi.pkl', 'rb') as f:
        for data in f:
            # print(data)
            client.send(data)
    f.close()
    print("Successfully send the cloud model!")
    client.close()
