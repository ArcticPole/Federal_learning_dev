# _*_ coding:utf-8 _*_
"""
Created on Wed Feb 2 11:41:00 2022

@author: Lin Ze

This file is used to send models.
"""
from socket import *
import _thread
import time

def tcplink(skt, addr):
    print(skt)
    print(addr, "connected...")
    print('Start sending')
    with open('test_samp.pth', 'rb') as f:
        for data in f:
            # print(data)
            skt.send(data)
    f.close()
    skt.close()
    print("Successfully coding!")
    return False

HOST = "127.0.0.1"
PORT = 23333
ADDR = (HOST, PORT)

server = socket(AF_INET, SOCK_STREAM)
server.bind(ADDR)
server.listen(5)
state = True
while state == True:
    print("Waiting for connection...")
    skt, addr = server.accept()
    print(skt)
    state = tcplink(skt, addr)
print("Local data sended!")

skt, addr = server.accept()
with open("local_model/model_from_cloud.pkl", "wb") as f:
    while True:
        data = skt.recv(1024)
        if not data:
            break
        f.write(data)
print("Cloud model recieved!")

