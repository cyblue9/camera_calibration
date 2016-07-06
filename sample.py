import cv2
import numpy as np
import glob

# termination criteria
critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

# prepare object points, like(0,0,0),(1,0,0),(2,0,0),.....,(6,5,0)
objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

images = glob.glob('./data/left[0-9]'+ '*.jpg') # image title
for i,fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret,corners = cv2.findChessboardCorners(gray,(7,6),None)

    # If found. add object points, image points(after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),critera)
        imgpoints.append(corners)

        # Draw and display the corners
        temp = cv2.drawChessboardCorners(img,(7,6),corners,ret)
        title = '{0}.jpg'.format(i)
        cv2.imwrite(title,temp)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration
ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

# refine the camera matrix
img = cv2.imread('left12.jpg')
h,w = img.shape[:2]
newcameramtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# Undistortion
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)

x,y,w,h = roi
dst = dst[y:y+h,x:x+w]

# Draw Line to check the camera Calibration
dst2 = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
ret,corners = cv2.findChessboardCorners(dst2,(7,6),None)
cv2.cornerSubPix(dst2,corners,(11,11),(-1,-1),critera)
cv2.line(dst,(corners[0][0][0],corners[0][0][1]),(corners[6][0][0],corners[6][0][1]),(0,0,255),2)
cv2.line(dst,(corners[0][0][0],corners[0][0][1]),(corners[35][0][0],corners[35][0][1]),(0,0,255),2)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()

# Re-projection Error
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2,__ = cv2.projectPoints(objpoints[i],rvecs[i],tvecs[i],mtx,dist)
    error = cv2.norm(imgpoints[i],imgpoints2,cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print "total error", mean_error/len(objpoints)
