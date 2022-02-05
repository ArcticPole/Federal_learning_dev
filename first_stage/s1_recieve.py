# _*_ utf-8 _*_
"""
Created on Wed Feb 2 11:41:00 2022

@author: Lin Ze

This file is used to receive models from other robotic arms in the same scene.
"""
from socket import *

def recieve_model(HOST):
    #HOST = "127.0.0.1"
    PORT = 23333
    ADDR = (HOST,PORT)
    client = socket(AF_INET,SOCK_STREAM)
    client.connect(ADDR)
    
    with open("./merge_model.pkl","wb") as f:
        while True:
            data = client.recv(1024)
            if not data:
                break
            f.write(data)
    
    f.close()
    print("Successfully got the weight data!")
    client.close()
    PATH = "./merge_model.pkl"
    return PATH