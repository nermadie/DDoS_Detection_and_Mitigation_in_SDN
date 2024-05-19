#!/bin/csh

# Iperf to server using udp
iperf -c 10.0.0.1 -u -b 100M -t 10 -i 1

