import cv2
from pymongo import MongoClient
import time
import datetime
import urllib 
import numpy as np

timestamp=time.time()
DB_IP='52.27.134.146'
CAM_URL='http://192.168.0.72:8080/?action=stream'

db_client=MongoClient(DB_IP, 27017)
db=db_client.power_stats
print(db.snapshot)
stream=urllib.urlopen(CAM_URL)
bytes=''
while True:
    bytes+=stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
#        cv2.imshow('RAW',img)
        src_img =  cv2.imread('meter.jpg',0)          # queryImage
        # Initiate STAR detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints with ORB
        kp1, des1 = sift.detectAndCompute(src_img,None)
        kp2, des2 = sift.detectAndCompute(img,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
        img3 = cv2.drawMatchesKnn(src_img,kp1,img,kp2,matches,None,**draw_params)
        cv2.imshow('keypoints',img3)
        # Find circles of known radius and draw them
        gray = (cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
#        equalized_gray = cv2.equalizeHist(gray)

        # Find most prominent lines and draw them
        edges = cv2.Canny(gray,25,100,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/200,180)
        avg_theta = 0
        for rho,theta in lines[0]:
            avg_theta += theta
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(230,255,2),1)

        # Fix image rotation
        #print(360*avg_theta/lines[0].size/2-90)
        #M = cv2.getRotationMatrix2D((640/2,480/2),360*avg_theta/lines[0].size/2-91.87,1)
        #img = cv2.warpAffine(img,M,(640,480))

        #cv2.imshow('Equalized',gray)
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,10,param1=25,param2=25,minRadius=30,maxRadius=45)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
#            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
#            cv2.circle(img,(i[0],i[1]),2,(0,0,255),1)
            # draw the brightest spot in this circle
            r = 10
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray[i[1]-r:i[1]+r, i[0]-r:i[0]+r])
            cv2.circle(img, (maxLoc[0]+i[0]-r,maxLoc[1]+i[1]-r), 2, (0, 255, 0), 2)
            cv2.circle(img, (maxLoc[0]+i[0]-r,maxLoc[1]+i[1]-r), i[2], (0,0,255),2)
            # take the avg_point between brightest_spot & center_of_circle as mathematical center 

        M = cv2.getRotationMatrix2D((640/2,480/2),360*avg_theta/lines[0].size/2-91.87,1)                                                                          
        img = cv2.warpAffine(img,M,(640,480))  
        cv2.imshow('gray,edge,hough,rotate',img)
        print(time.time()-timestamp)
	timestamp=time.time()
	if cv2.waitKey(1) ==27:
            exit(0)   
