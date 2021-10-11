import cv2 as cv
import numpy as np
import pickle
import pathlib
from collections.abc import Iterable
from typing import Union, List, Tuple

import cam_calib_v2

MARKER_IDS = [89,28,3,184]


def __save_markers(mark_dict, marker_id, marker_size=600, marker_folder = './resources/markers'):
    if hasattr(marker_id, '__iter__') is False:
        marker_id = [marker_id]
    
    for id in marker_id:
        marker_img = cv.aruco.drawMarker(aruco_mark_dict, id, marker_size)
        cv.imwrite(marker_folder + '\\marker_{}.jpg'.format(id), marker_img)

def __detect_markers(mark_dict, img, params = None):
    if params is None:
        params = cv.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(img, mark_dict, parameters=params)
    return markerCorners, markerIds, rejectedCandidates

def __draw_detected_markers(img, corners, ids, rejected=None):
    for corn in corners:
        for idx, _ in enumerate(corn[0]):
            cv.line(img, corn[0, idx - 1].astype(int), corn[0, idx].astype(int), (0,255,0), thickness=2)
        
        cv.circle(img, corn[0,1].astype(int), 5, (0,0,255), thickness=-1)

# def get_rvec_tvec():
#     pass

def __get_WLP_aruco_board(aruco_dict):
    obj_points = np.array([
        [[0,0,0], [0,50,0], [50,50,0], [50,0,0]],
        [[0,130,0], [0,180,0], [50,180,0], [50,130,0]],
        [[217,0,0], [217,50,0], [267,50,0], [267,0,0]],
        [[217,130,0], [217,180,0], [267,180,0], [267,130,0]]
        ]).astype(np.float32)

    board = cv.aruco.Board_create(obj_points, aruco_dict, np.array(MARKER_IDS))
    return board


# def load_calib(path='./resources/', app_string=''):
#     with open(path + 'cam_calib' + app_string + '.pkl', 'rb') as f:
#         var = pickle.load(f)
#     return np.array(var[0]), np.array(var[1])    

def __draw_tr_vec(img, tvec, rvec):
    # s = f'{tvec[0,0]:.2f}'
    cv.putText(img, f'{tvec[0,0]:.2f} {tvec[1,0]:.2f} {tvec[2,0]:.2f}', (10,1070), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv.putText(img, f'{rvec[0,0]:.2f} {rvec[1,0]:.2f} {rvec[2,0]:.2f}', (10,1020), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return img

def __save_extrinsics(rvec, tvec, path='./resources/', app_string='ext_'):
    var = [rvec, tvec]
    with open(path + app_string + '.pkl', 'wb') as f:
        pickle.dump(var, f)

def load_extrinsics(path='./resources/', app_string='ext_'):
    with open(path + app_string + '.pkl', 'rb') as f:
        var = pickle.load(f)
    return np.array(var[0]), np.array(var[1])


def __calc_extrinsic(image_paths: Union[Tuple, List], save_test_img: bool=False):
    # Make sure it is iterable to enable looping
    if not isinstance(image_paths, Iterable):
        raise TypeError

    # Load aruco variables
    aruco_mark_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    aruco_board = __get_WLP_aruco_board(aruco_mark_dict)

    # Load camera intrinsic parameters
    mtx, dist = cam_calib_v2.load_calib()

    rvecs = []
    tvers = []
    # Calculate extrinsic parameters
    for img_path in image_paths:
        # Load image
        img = cv.imread('.\\resources\\img\\test_exp_1_z62\\DSC_3085.JPG')
        
        # Detect Markers
        markerCorners, markerIds, rejectedCandidates = __detect_markers(aruco_mark_dict, img)
        __draw_detected_markers(img, markerCorners, markerIds)

        if len(markerCorners) > 0:
            retval, rvec, tvec = cv.aruco.estimatePoseBoard(markerCorners, markerIds, aruco_board, mtx, dist, None, None)
            rvecs.append(rvec)
            tvers.append(tvec)
            # rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners, 50, mtx, dist)

            # Draw Marker Points
            cv.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
            # for rvec, tvec in zip(rvecs, tvecs):
            cv.aruco.drawAxis(img, mtx, dist, rvec, tvec, 50)

            if save_test_img:
                __draw_tr_vec(img, tvec, rvec)

            small_img = cv.resize(img, (0,0), fx = 0.25, fy = 0.25)
            cv.imshow('', small_img)

    return rvecs, tvecs

def walp_get_marker(img_path: str, extrinsics_path: str, extrinsics_app_string: str='ext_', cam_calib_path: str='./resources/', cam_calib_app_string: str='', save_debug_img: bool=False, path_debug_img: str='./debug_imgs/'):
    # Get marker information
    aruco_mark_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    aruco_board = __get_WLP_aruco_board(aruco_mark_dict)

    # Load camera calibration
    mtx, dist = cam_calib_v2.load_calib(path=cam_calib_path, app_string=cam_calib_app_string)

    # Load image
    img = cv.imread(img_path)

    # Detect Markers
    markerCorners, markerIds, rejectedCandidates = __detect_markers(aruco_mark_dict, img)
    __draw_detected_markers(img, markerCorners, markerIds)


    if len(markerCorners) > 0:
        retval, rvec, tvec = cv.aruco.estimatePoseBoard(markerCorners, markerIds, aruco_board, mtx, dist, None, None)
        # rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners, 50, mtx, dist)

        if save_debug_img:
            # Make sure debug folder exists
            path = pathlib.Path(path_debug_img)
            path.mkdir(parents=True, exist_ok=True)

            # Draw Marker Points
            cv.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
            cv.aruco.drawAxis(img, mtx, dist, rvec, tvec, 50)
            __draw_tr_vec(img, tvec, rvec)

            # small_img = cv.resize(img, (0,0), fx = 0.25, fy = 0.25)
            img = cv.undistort(img, mtx, dist)


            img_path.split('\\')
            cv.imwrite(path_debug_img + 'mark_detect_' + img_path.split('\\')[-1], img)

        
        path = pathlib.Path(extrinsics_path)
        path.mkdir(parents=True, exist_ok=True)
        __save_extrinsics(rvec, tvec, path=extrinsics_path, app_string=extrinsics_app_string + img_path.split('\\')[-1].split('.')[0])



if __name__ == '__main__':
    img_path = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec\\DSC_3364.jpg'
    walp_get_marker(img_path, './ext_res/', save_debug_img=True)



    # aruco_mark_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    # aruco_board = __get_WLP_aruco_board(aruco_mark_dict)

    # # Generate and Save markers
    # # save_markers(aruco_mark_dict, MARKER_IDS)
    # # inVid = cv.VideoCapture()
    # # inVid.open(0)
    # # inVid.set(3, 1920)
    # # inVid.set(4, 1080)
    # # while (inVid.grab()):
    # while True:
    #     # img = cv.imread('.\\resources\\img\\test_exp_1_z62\\DSC_3085.JPG')
    #     img = cv.imread('F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec\\DSC_3364.jpg')
    #     # retval, img = inVid.retrieve()
        
    #     # Detect Markers
    #     markerCorners, markerIds, rejectedCandidates = __detect_markers(aruco_mark_dict, img)
    #     __draw_detected_markers(img, markerCorners, markerIds)

    #     mtx, dist = cam_calib_v2.load_calib()
        
    #     if len(markerCorners) > 0:
    #         retval, rvec, tvec = cv.aruco.estimatePoseBoard(markerCorners, markerIds, aruco_board, mtx, dist, None, None)
    #         # rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners, 50, mtx, dist)

    #         # Draw Marker Points
    #         cv.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
    #         # for rvec, tvec in zip(rvecs, tvecs):
    #         cv.aruco.drawAxis(img, mtx, dist, rvec, tvec, 50)
    #         __draw_tr_vec(img, tvec, rvec)

    #     small_img = cv.resize(img, (0,0), fx = 0.25, fy = 0.25)
    #     cv.imshow('', small_img)
    #     retval = cv.waitKey(0)
    #     if retval != -1:
    #         __save_extrinsics(rvec, tvec)
    #         img = cv.undistort(img, mtx, dist)
    #         cv.imwrite('./resources/test_img.jpg', img)
    #         exit()