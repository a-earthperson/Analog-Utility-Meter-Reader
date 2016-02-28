# Analog-Utility-Meter-Reader
An OpenCV implementation of an Analog Utility Meter Reader. Image processing code written in Python, server-side visualization implemented using Node JS and MongoDB.

The project is divided into three separate modules, each of which are designed to communicate over LAN and/or WAN connections. 

Topology
===========
    1. MJPG streaming Server : Implemented using a USB webcam connected to a Raspberry Pi. Stream is exposed over LAN by an HTTP server.
    2. Image Recognition     : Performed on a local LAN host, preferably with higher compute capability than the Raspberry Pi. 
    3. DB & Web server       : Remote host, preferably a VPS, capable of handling multiple simultaneous connections.

Disclaimer
------------
This is an extremely bare-bones, hackish implementaton. Majority of the code has been borrowed from the OpenCV documentation, with very minimal changes.
