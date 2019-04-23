#!/bin/bash

IP=192.168.111.1 #`hostname --ip-address`
PORT=31000
echo $IP:$PORT
mongod --dbpath ./db --bind_ip $IP --port $PORT --noprealloc
