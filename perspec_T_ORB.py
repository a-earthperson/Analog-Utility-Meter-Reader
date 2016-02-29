import cv2
from pymongo import MongoClient
import time
import math
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
        #cv2.setMouseCallback('RAW', mouse_callback)
        #cv2.setMouseCallback('REF', mouse_callback)
        #cv2.imshow('RAW',img)
        #cv2.imshow('REF',img_ref)        

        #ORB to get corresponding points
        kp, des = orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        m = bf.match(des_ref,des)
        m = sorted(m, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img_ref,kp_ref,img,kp,m[:4], None,flags=2)
        cv2.imshow('Matches',img3)
            
#        pts_ref = np.float32([[kp_ref[m[0].queryIdx].pt[0],kp_ref[m[0].queryIdx].pt[1]],[kp_ref[m[1].queryIdx].pt[0],kp_ref[m[1].queryIdx].pt[1]],[kp_ref[m[2].queryIdx].pt[0],kp_ref[m[2].queryIdx].pt[1]],[kp_ref[m[3].queryIdx].pt[0],kp_ref[m[3].queryIdx].pt[1]]])
#        pts     = np.float32([[kp[m[0].trainIdx].pt[0],kp[m[0].trainIdx].pt[1]],[kp[m[1].trainIdx].pt[0],kp[m[1].trainIdx].pt[1]],[kp[m[2].trainIdx].pt[0],kp[m[2].trainIdx].pt[1]],[kp[m[3].trainIdx].pt[0],kp[m[3].trainIdx].pt[1]]])

        # Perspective Transform
        pts_ref = np.float32([[223,149],[441,149],[481,314],[179,309]])
        pts     = np.float32([[165,388],[185,103],[378,86],[353,435]])
        M = cv2.getPerspectiveTransform(pts,pts_ref)
        dst = cv2.warpPerspective(img,M,(cols,rows))
        #cv2.imshow('Perspective Transform',dst)

        try:
            # Find circles
            gray = cv2.equalizeHist(cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY))
            #gray = cv2.medianBlur(gray,5)
            #gray = cv2.Canny(gray,10,20)
            #cv2.imshow('Edge Detect',gray)
            circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=25,minRadius=25,maxRadius=35)
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                #draw the outer circle 
                #cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle 
                #cv2.circle(img,(i[0],i[1]),2,(0,0,255),1)
                
                #Find brigthest spot in this circle
                r = 5
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray[i[1]-r:i[1]+r, i[0]-r:i[0]+r])
                # Draw center of brightest spot in circle
                cv2.circle(dst, (maxLoc[0]+i[0]-r,maxLoc[1]+i[1]-r), 2, (0, 255, 0), 1)
                cv2.circle(dst, (maxLoc[0]+i[0]-r,maxLoc[1]+i[1]-r), i[2], (0,0,255),1)

                # Find most prominent line within small_circle
                small_circle = gray[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
                try:
                    edges = cv2.Canny(small_circle,25,100,apertureSize = 3)
                    cv2.imshow('edges',edges)
                    minLineLength = 20
                    maxLineGap = 2
                    lines = cv2.HoughLinesP(edges,1,np.pi/360,5,minLineLength,maxLineGap)
                    for x1,y1,x2,y2 in lines[0]:
                        cv2.line(small_circle,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.imshow('small_circle',small_circle)
                except:
                    print('No lines in circle detected')
                    pass
        except:
            print('No circles detected')
            pass
         
        cv2.imshow('Perspective Transform',dst)
        # Print per-frame processing delay
        print(time.time()-ts)
        ts=time.time()

        if cv2.waitKey(1) == 27:
            exit(0)

def mouse_callback(event,x,y,flags,param):
        print x, y
main()
