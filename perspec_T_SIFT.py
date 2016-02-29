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
sift = cv2.xfeatures2d.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(img_ref,None)

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
        kp, des = sift.detectAndCompute(img,None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_ref,des,k=2)
        m = []
        for ma,na in matches:
            if ma.distance < 0.75*na.distance:
                m.append([ma])
        img3 = cv2.drawMatchesKnn(img_ref,kp_ref,img,kp,m[:4], None,flags=2)
        cv2.imshow('MatchesKnn',img3)

        #pts_ref = np.float32([[kp_ref[m[0].queryIdx].pt[0],kp_ref[m[0].queryIdx].pt[1]],[kp_ref[m[1].queryIdx].pt[0],kp_ref[m[1].queryIdx].pt[1]],[kp_ref[m[2].queryIdx].pt[0],kp_ref[m[2].queryIdx].pt[1]],[kp_ref[m[3].queryIdx].pt[0],kp_ref[m[3].queryIdx].pt[1]]])
        #pts     = np.float32([[kp[m[0].trainIdx].pt[0],kp[m[0].trainIdx].pt[1]],[kp[m[1].trainIdx].pt[0],kp[m[1].trainIdx].pt[1]],[kp[m[2].trainIdx].pt[0],kp[m[2].trainIdx].pt[1]],[kp[m[3].trainIdx].pt[0],kp[m[3].trainIdx].pt[1]]])
        # Perspective Transform
        #M = cv2.getPerspectiveTransform(pts_ref,pts)
        #dst = cv2.warpPerspective(img,M,(cols,rows))
        #cv2.imshow('Perspective Transform',dst)

        # Print lag
        print(time.time()-ts)
        ts=time.time()

        if cv2.waitKey(1) == 27:
            exit(0)

main()
