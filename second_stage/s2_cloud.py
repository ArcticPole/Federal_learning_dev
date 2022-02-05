# _*_ coding:utf-8 _*_
"""
Created on Wed Feb 2 11:41:00 2022

@author: Lin Ze

This file is used to send models.
"""
from socket import *
import _thread


def tcplink(skt, addr):
    print(skt)
    print(addr, "connected...")
    print('Start sending')
    with open('../global_models/net_xiao_global_para_hy_2.pkl', 'rb') as f:
        for data in f:
            # print(data)
            skt.send(data)
    f.close()
    skt.close()

    print("Successfully!")


HOST = "127.0.0.1"
PORT = 23333
ADDR = (HOST, PORT)

server = socket(AF_INET, SOCK_STREAM)
server.bind(ADDR)
server.listen(5)
while True:
    print("Waiting for connection...")
    skt, addr = server.accept()
    print(skt)
    try:
        _thread.start_new_thread(tcplink, (skt, addr))
    except KeyboardInterrupt:
        break

server.close()