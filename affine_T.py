import cv2
from pymongo import MongoClient
import time
import datetime
import urllib 
import numpy as np

# Initialize Constants
DB_IP='52.27.134.146'
CAM_URL='http://192.168.0.72:8080/?action=stream'
cols=640
rows=480

# setup DB 
db_client=MongoClient(DB_IP, 27017)
db=db_client.power_stats
print(db.snapshot)

# setup helper OpenCV functions
img_ref=cv2.imread('iphone.jpg',0) #queryImage
orb = cv2.ORB_create()
kp_ref, des_ref = orb.detectAndCompute(img_ref,None)

def main():
    stream=urllib.urlopen(CAM_URL)
    bytes=''
    ts=time.time()
    while True:
        bytes+=stream.read(2048)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a==-1 or b==-1:
            continue

        # Frame available
        rtimestamp=time.time()
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        cv2.imshow('RAW',img)
        
        #ORB to get corresponding points
        kp, des = orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_ref,des)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img_ref,kp_ref,img,kp,matches[:4], None,flags=2)
        cv2.imshow('Matches',img3)

#        pts_src = np.float32([[kp_ref[0].pt[0],kp_ref[0].pt[1]],[kp_ref[1].pt[0],kp_ref[1].pt[1]],[kp_ref[0].pt[0],kp_ref[0].pt[1]],[kp_ref[0].pt[0],kp_ref[0].pt[1]]
        # Perspective Transform
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        Tr_M = cv2.getAffineTransform(pts1,pts2)
        oimg = cv2.warpAffine(img,Tr_M,(cols,rows))
        cv2.imshow('Perspective Transform',oimg)

        # Print lag
        print(time.time()-ts)
        ts=time.time()

        if cv2.waitKey(1) == 27:
            exit(0)

main()
