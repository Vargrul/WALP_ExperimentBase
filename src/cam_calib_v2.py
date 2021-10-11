
#!/usr/bin/env python
import argparse

import cv2
import numpy as np
import os
import glob
from enum import Enum, auto
import pickle

class GridType(Enum):
    CHECKERBOARD = auto()
    ASYMM_CIRCLES = auto()

class InputType(Enum):
    CAMERA = auto()
    IMAGE_FILES = auto()

def __get_obj_points(grid_type: GridType, grid_size):
    grid_scale = __get_grid_scale(grid_type)
    if grid_type is GridType.CHECKERBOARD:
        # Defining the world coordinates for 3D points
        # TODO Add scale
        objp = np.zeros((1, grid_size[0] * grid_size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)*grid_scale
    elif grid_type is GridType.ASYMM_CIRCLES:
        # Defining the world coordinates for 3D points
        objectPoints = []
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                objectPoints.append( ((2*j + i%2)*grid_scale, i*grid_scale, 0) )

        objp = np.array(objectPoints).astype(np.float32)

    return objp

def __get_criteria(grid_type: GridType):
    if grid_type is GridType.CHECKERBOARD:
        params = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    elif grid_type is GridType.ASYMM_CIRCLES:
        params = cv2.SimpleBlobDetector_Params()
            
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2000
        params.maxArea = 2400000

        params.minDistBetweenBlobs = 20

        params.filterByColor = True
        params.filterByConvexity = False

        # tweak these as you see fit
        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.2

        # # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = True
        # params.filterByInertia = False
        params.minInertiaRatio = 0.01

    return params

def __get_grid_size(grid_type: GridType):
    if grid_type is GridType.CHECKERBOARD:
        grid_size = (6,9)
    elif grid_type is GridType.ASYMM_CIRCLES:
        grid_size = (4,11)

    return grid_size

def __get_grid_scale(grid_type: GridType):
    if grid_type is GridType.CHECKERBOARD:
        grid_scale = 25
    elif grid_type is GridType.ASYMM_CIRCLES:
        grid_scale = 20

    return grid_scale

def __find_pattern(img, grid_type: GridType, grid_size):
    criteria = __get_criteria(grid_type)
    if grid_type is GridType.CHECKERBOARD:
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(img, grid_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    elif grid_type is GridType.ASYMM_CIRCLES:
        # Find the chess board corners
        detector = cv2.SimpleBlobDetector_create(criteria)
        blop_img = img.copy()
        key_pnt = detector.detect(blop_img)
        for kp in key_pnt:
            blop_img = cv2.circle(blop_img, np.array(kp.pt).astype(np.uint32), 100, (255,0,0))
        
        cv2.imshow('blops', blop_img)
        cv2.waitKey(0)

        circle_params = cv2.CirclesGridFinderParameters()
        circle_params.kmeansAttempts = 1000
        
        ret, corners = cv2.findCirclesGrid(img, grid_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=detector, parameters=circle_params)
    
    
    # If desired number of corner are detected,
    # we refine the pixel coordinates and display 
    # them on the images of checker board
    corners2 = None
    if ret == True:        
        if grid_type is GridType.ASYMM_CIRCLES:
            # No refinement as there is no conors on a circle.
            corners2 = corners
        elif grid_type is GridType.CHECKERBOARD:
            # refining pixel coordinates for given 2d points.
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

    return ret, corners2


def __calib_cam(grid_type: GridType, input_type: InputType, folder_path:str=None):

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 

    # Defining the dimensions of checkerboard
    grid_size = __get_grid_size(grid_type)

    objp = __get_obj_points(grid_type, grid_size)

    if input_type is InputType.IMAGE_FILES:
        print(f'Using files for calibration.')
        # Extracting path of individual image stored in a given directory
        frame = glob.glob(folder_path)
        file_amount = len(frame)
        current_file_nr = 0
        print(f'Found {file_amount} image files for calibration.')

    elif input_type is InputType.CAMERA:
        # Open camera
        cap = cv2.VideoCapture(0)
        # Set Width and Size
        cap.set(3, 1920)
        cap.set(4, 1080)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

    while True:
        if input_type is InputType.CAMERA:
            ret, img = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        elif input_type is InputType.IMAGE_FILES:
            if current_file_nr < file_amount:
                print(f'Proccessing Image {current_file_nr} of {file_amount}')
                img = cv2.imread(frame[current_file_nr])
                current_file_nr = current_file_nr + 1
            else:
                break
        
        # Convert to grayscale
        gray = img
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners2 = __find_pattern(gray, grid_type, grid_size)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, grid_size, corners2, ret)
        
        cv2.imshow('img',img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    img_size = (gray.shape[1], gray.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print("Ret Val : \n")
    print(ret)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    return ret, mtx, dist, rvecs, tvecs


def __draw_axis(img, imgpts, linewidth=2):
    imgpts = imgpts.astype(np.uint)
    center = tuple(imgpts[0].ravel())
    img = cv2.line(img, center, tuple(imgpts[1].ravel()), (255,0,0), linewidth)
    img = cv2.line(img, center, tuple(imgpts[2].ravel()), (0,255,0), linewidth)
    img = cv2.line(img, center, tuple(imgpts[3].ravel()), (0,0,255), linewidth)
    return img

def __get_axis_points(length=25):
    axis = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    return axis

def world_to_view():
    pass

def __show_test_img(grid_type: GridType, c_mtx, dist_cof, folder_path, scale=1):
    frame = glob.glob(folder_path)

    grid_size = __get_grid_size(grid_type)

    objp = __get_obj_points(grid_type, grid_size)

    img = cv2.imread(frame[0])

    # Convert to grayscale
    gray = img
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners2 = __find_pattern(gray, grid_type, grid_size)

    if ret:
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, c_mtx, dist_cof)
        var = [rvecs, tvecs]
        with open('./resources/extrinsic_calib_test' + '.pkl', 'wb') as f:
            pickle.dump(var, f)

        img = cv2.drawFrameAxes(img, c_mtx, dist_cof, rvecs, tvecs, 50)
        img = cv2.undistort(img, c_mtx, dist_cof)
        cv2.imwrite('./resources/calib_test_img.jpg', img)

        # axis_points = get_axis_points()
        # axis_points, jac = cv2.projectPoints(axis_points, rvecs, tvecs, c_mtx, dist_cof)
        # img = draw_axis(img, axis_points)
    
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    cv2.imshow('img',img)
    cv2.waitKey(0)

def __run_test_stream(grid_type: GridType, c_mtx, dist_cof):
    cap = cv2.VideoCapture(0)
    # Set Width and Size
    cap.set(3, 1920)
    cap.set(4, 1080)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    grid_size = __get_grid_size(grid_type)

    objp = __get_obj_points(grid_type, grid_size)

    while True:
        ret, img = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert to grayscale
        gray = img
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners2 = __find_pattern(gray, grid_type, grid_size)

        if ret:
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, c_mtx, dist_cof)

            axis_points = __get_axis_points()
            axis_points, jac = cv2.projectPoints(axis_points, rvecs, tvecs, c_mtx, dist_cof)
            img = __draw_axis(img, axis_points)
        
        cv2.imshow('img',img)
        cv2.waitKey(1)


def __save_calib(mtx, dist, path='./resources/', app_string=''):
    var = [mtx, dist]
    with open(path + 'cam_calib' + app_string + '.pkl', 'wb') as f:
        pickle.dump(var, f)

def load_calib(path='./resources/', app_string=''):
    with open(path + 'cam_calib' + app_string + '.pkl', 'rb') as f:
        var = pickle.load(f)
    return var[0], var[1]

def __create_parser():
    parser = argparse.ArgumentParser(description="Python Camera Calibration.")

    # Grid Type
    parser.add_argument('-g', '--grid', choices=['checkerboard', 'asymmetric_circles'], required=True)

    # Image Source
    parser.add_argument('-s', '--source', choices=['camera', 'folder'], required=True, help="The source of the images.")
    parser.add_argument('-f', '--folder', help="The given path to a folder containing image files.")
    parser.add_argument('-o', '--output', help='Defines the output file name. If not defined, the file will be named: cam_calib', default='')

    return parser

if __name__ == "__main__":
    parser = __create_parser()
    args = parser.parse_args()

    if args.grid == 'checkerboard':
        grid_type = GridType.CHECKERBOARD
    elif args.grid == 'asymmetric_circles':
        grid_type = GridType.ASYMM_CIRCLES
    
    
    if args.source == 'camera':
        input_type = InputType.CAMERA
    elif args.source == 'folder':
        input_type = InputType.IMAGE_FILES

    print(f'========== Calibrating Camera ==========')

    ret, mtx, dist, rvecs, tvecs = __calib_cam(grid_type, input_type, args.folder)
    # './resources/img/cam_cal_test/*.jpg'

    __save_calib(mtx, dist, app_string=args.output)

    mtx2, dist2 = load_calib(app_string=args.output)

    # run test stream
    # run_test_stream(grid_type, mtx2, dist2)

    __show_test_img(grid_type, mtx2, dist2, args.folder, scale=0.25)
