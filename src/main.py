import glob

import marker_detection
import create_max_script
import os
from tqdm import tqdm


def main(path_img_folder: str, ext_folder: str='ext_res\\', sensor_size=[36,24], render_path='./render_out/', cam_calib_app_string='', single_cam_pas=False):
    if render_path[0] == '.':
        render_path = os.getcwd() + render_path[1:]
    
    # Get all images
    files = glob.glob(path_img_folder + 'p\\*.jpg')

    # Get extrinsics for all images
    for img_path in tqdm(files, ncols=100, desc='Marker Detection'):
        marker_detection.walp_get_marker(img_path, path_img_folder + ext_folder, save_debug_img=True, cam_calib_app_string=cam_calib_app_string, path_debug_img=path_img_folder+'debug_imgs\\')

    # generate Max Script
    # ext_files = glob.glob(ext_folder)
    create_max_script.walp_generate_max_script(files, sensor_size, render_path, extrin_path=path_img_folder + ext_folder, single_cam_pos=single_cam_pas)

if __name__ == "__main__":
    # path_img = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec\\'
    # path_render = 'D:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec\\'
    # single_cam_pas = True
    # path_img = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\Reconstruction_bonus_experiment\\1sec\\'
    # path_render = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\Reconstruction_bonus_experiment\\1sec\\'
    # single_cam_pas = True
    path_img = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\1_sec_cloud_changing\\'
    path_render = path_img
    single_cam_pas = True
    # path_img = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\15_sec_chaning_cloud_long\\'
    # path_render = path_img
    # single_cam_pas = False

    cam_calib_app_string = 'z60_70mm'

    main(path_img, render_path=path_render, cam_calib_app_string=cam_calib_app_string, single_cam_pas = single_cam_pas)