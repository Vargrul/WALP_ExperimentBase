import marker_detection
import cam_calib_v2
import numpy as np
import cv2 as cv
import pathlib
from exif import Image
from datetime import datetime, date, timedelta
from collections.abc import Iterable
from typing import Tuple, List
import glob

def ms_render_setup(img_size, render_path, render_file_name, output_raw=True, output_split=True):
    # get render
    #   r = renderers.current
    st = f'global r = renderers.current\n'
    st = st + f'\n'

    # render size
    #   .output_width : integer
    #   .output_height : integer
    st = st + f'r.output_width = {img_size[0]}\n'
    st = st + f'r.output_height = {img_size[1]}\n'
    st = st + f'renderWidth = {img_size[0]}\n'
    st = st + f'renderHeight = {img_size[1]}\n'
    st = st + f'\n'

    # Normal file output
    split_path = render_path.replace('\\', '/')
    st = st + f'rendSaveFile  = true\n'
    st = st + f'rendOutputFilename = "{split_path + render_file_name}.jpg"\n'
    

    # enable raw file, and 
    #   .output_saveRawFile : boolean
    #   .output_rawFileName : filename
    if output_raw:
        exr_path = render_path + 'vray_raw/'
        exr_path = exr_path.replace('\\', '/')   #This is due to some 3ds max weiredness...
        st = st + f'r.output_saveRawFile = true\n'
        st = st + f'r.output_rawFileName = "{exr_path + render_file_name}.exr"\n'
        st = st + f'\n'
    else:
        st = st + f'r.output_saveRawFile = false\n'

    # change separete render channels path
    if output_split:
        st = st + f'r.output_splitgbuffer = true\n'
        split_path = render_path.replace('\\', '/')
        st = st + f'bm_filename = "{split_path + render_file_name}.jpg"\n'
        st = st + f'bm=Bitmap 10 10 fileName: bm_filename\n'
        st = st + f'save bm\n'
        st = st + f'close bm\n'
        st = st + f'r.output_splitFileName = bm\n'
        st = st + f'deleteFile bm_filename\n'
        st = st + f'\n'
    else:
        st = st + f'r.output_splitgbuffer = false\n'

    return st

def ms_cam_setup(cam_mtx, tvec, rvec, camera_name: str, render_path: str, img_size: Tuple[int, int], sensor_size: Tuple[float, float]) -> str:
    fovx, fovy, focal_length, principalPoint, aspectRatio = cv.calibrationMatrixValues(cam_mtx, img_size, sensor_size[0], sensor_size[1])

    # scale for 3ds max wiredness...
    tvec = tvec / 10    

    rot_mat, _ = cv.Rodrigues(rvec)
    inv_rot_mat = np.transpose(rot_mat)
    inv_tvec = np.dot(-inv_rot_mat, tvec)

    # Convert from OpenCV to 3ds max (OpenGL) coordinate system
    mat = [[1, 0, 0], 
           [0, -1, 0], 
           [0, 0, -1]]
    inv_rot_mat = inv_rot_mat @ mat

    st = f'global camName = "{camera_name}"\n'
    st = st + f'\n'
    st = st + f'cam = getNodeByName(camName)\n'
    st = st + f'\n'
    st = st + f'cam.lens_breathing_amount = 0.0\n'
    st = st + f'cam.focal_length_mm = {focal_length}\n'
    st = st + f'cam.film_preset = "Custom"\n'
    st = st + f'cam.film_width_mm = {sensor_size[0]}\n'
    st = st + f'\n'
    st = st + f'cam.targeted = false\n'
    st = st + f'cam.rotation = quat 0 0 0 0\n'
    st = st + f'\n'
    st = st + f'camTrans = cam.transform'
    st = st + f'\n'    
    st = st + f'camTrans.row1 = [{inv_rot_mat[0,0]}, {inv_rot_mat[1,0]}, {inv_rot_mat[2,0]}]\n'
    st = st + f'camTrans.row2 = [{inv_rot_mat[0,1]}, {inv_rot_mat[1,1]}, {inv_rot_mat[2,1]}]\n'
    st = st + f'camTrans.row3 = [{inv_rot_mat[0,2]}, {inv_rot_mat[1,2]}, {inv_rot_mat[2,2]}]\n'
    st = st + f'camTrans.row4 = [{inv_tvec[0,0]}, {inv_tvec[1,0]}, {inv_tvec[2,0]}]\n'
    st = st + f'\n'
    st = st + f'cam.transform = camTrans'
    st = st + f'\n'

    return st

def ms_setup_daylight(daylight_name: str, dt: datetime, gps_data: Tuple[float, float]) -> str:
    # get daylight system
    st = f'd = getNodeByName("{daylight_name}")\n'

    delta_days = get_solar_date_value(dt)
    f_time = get_solar_time_value(dt)
    gps_floats = (get_GPS_single_value(gps_data[0]), get_GPS_single_value(gps_data[1]))

    # https://help.autodesk.com/view/MAXDEV/2021/ENU/?guid=GUID-9EEFF581-95CC-4D3C-887F-A7E1B2752C4F
    st = st + f'd.controller.Solar_Time.controller.value = {f_time}\n'
    st = st + f'd.controller.Solar_Date.controller.value = {delta_days}\n'
    st = st + f'\n'

    # Setup Lat Long for daylight
    st = st + f'd.controller.Latitude = {gps_floats[0]}\n'
    st = st + f'd.controller.Longitude = {gps_floats[1]}\n'
    st = st + f'\n'

    return st

def ms_render(render_path: str=None, save_vfb_img: bool=False) -> str:
    if render_path is not None and not save_vfb_img:
        st = f'render camera:cam outputfile:"{render_path}" vfb:false cancelled:&wasCancelled \n'
    else:
        st = f'render camera:cam vfb:false cancelled:&wasCancelled\n'
    st = st + f'if wasCancelled do\n'
    st = st + f'(\n'
    st = st + f'\treturn false\n'
    st = st + f')\n'

    if save_vfb_img:
        st = st + f'\n'    
        st = st + f'vfbControl #setchannel 0\n'
        st = st + f'vfbControl #saveimage "{render_path}"\n'

    st = st + f'\n'
    st = st + f'\n'
    return st

def ms_sun_vec(render_path: str, sun_name: str='Sun001') -> str:
    sun_vec_file_name = '/sun_vec/sun_vec.txt'

    st = f'-- Sun Vector Output\n'
    st = st + f'sun = getNodeByName("{sun_name}")\n'
    st = st + f'outSunFile = openfile "{render_path + sun_vec_file_name}" mode:"at"\n'
    st = st + f'print sun.pos to:outSunFile\n'
    st = st + f'close outSunFile\n'
    st = st + f'\n'
    st = st + f'\n'

    return st

def ms_cam_vec(render_path: str) -> str:
    cam_vec_file_name = '/cam_vec/cam_vec.txt'

    st = f'-- Cam Vector Output\n'
    st = st + f'outCamFile = openfile "{render_path + cam_vec_file_name}" mode:"at"\n'
    st = st + f'print cam.pos to:outCamFile\n'
    st = st + f'close outCamFile\n'
    st = st + f'\n'
    st = st + f'\n'

    return st

def ms_sky_vis(object_names: Tuple, render_path: str, sky_vis_light_name: str='light_SV') -> str:
    skyvis_path = 'sky_vis/'
    skyvis_file_name = 'sky_vis'
    
    # init str
    st = f'-- ===================================================\n'
    st = st + f'-- ============= SKY VIS RENDER ======================\n'
    st = st + f'-- ===================================================\n'
    st = st + f'\n'

    # create new mat
    st = st + f'sky_vis_mat = VRayMtl()\n'
    st = st + f'sky_vis_mat.diffuse = color 255 255 255\n'
    st = st + f'\n'
    

    for obj_name in object_names:
        # Get object as variables
        st = st + f'var_{obj_name} = getNodeByName("{obj_name}")\n'

        # Save old mats
        st = st + f'var_{obj_name}_old_mat = var_{obj_name}.material\n'

        # Apply new mat
        st = st + f'var_{obj_name}.material = sky_vis_mat\n'
        st = st + f'\n'

    # Change lighting to "sky vis lighting"
    st = st + f'd.sun.enabled = false\n'
    st = st + f'd.sky.on = false\n'
    st = st + f'sky_vis_light = getNodeByName("{sky_vis_light_name}")\n'
    st = st + f'sky_vis_light.on = true\n'
    st = st + f'\n'

    # Render skyvis
    # st = st + f'rendSaveFile = true\n'
    # st = st + f'rendOutputFilename = "{render_path + skyvis_path + skyvis_file_name}.jpg"\n'
    # exr_path = render_path + 'vray_raw/'
    exr_path = render_path + skyvis_path + 'vray_raw/'
    exr_path = exr_path.replace('\\', '/')   #This is due to some 3ds max weiredness...
    st = st + f'r.output_saveRawFile = True\n'
    st = st + f'r.output_rawFileName = "{exr_path + skyvis_file_name}.exr"\n'
    st = st + f'\n'
    st = st + ms_render()

    st = st + f'vfbControl #setchannel 0\n'
    st = st + f'vfbControl #saveimage "{render_path + skyvis_path + skyvis_file_name}.jpg"\n'

    # return render settings
    # st = st + f'rendSaveFile = false\n'
    # st = st + f'rendOutputFilename = ""\n'
    st = st + f'\n'

    # Set old mats
    for obj_name in object_names:
        st = st + f'var_{obj_name}.material = var_{obj_name}_old_mat\n'

    # Return to old lighting
    st = st + f'd.sun.enabled = true\n'
    st = st + f'd.sky.on = true\n'
    st = st + f'sky_vis_light.on = false\n'    
    
    st = st + f'\n'
    st = st + f'\n'
    return st

def ms_start() -> str:
    st = f'fn main = (\n'

    # Need to close the render window for render size to take effect
    st = st + f'renderSceneDialog.close()\n'
    st = st + f'\n'
    st = st + f'\n'
    return st

def ms_end() -> str:
    # Open the render window again
    st = f'renderSceneDialog.open()\n'
    st = st + f'\n'
    st = st + f')\n'
    st = st + f'main()\n'
    st = st + f'\n'

    return st

def get_solar_date_value(dt:datetime) -> int:
    dt_base = date(2021, 6, 21)
    delta_time = dt.date() - dt_base
    return delta_time.days

def get_solar_time_value(dt:datetime) -> float:
    f_time = dt.hour
    f_time = f_time + dt.minute/60
    f_time = f_time + dt.second/3600
    return f_time

def get_GPS_single_value(in_gps: List) -> float:
    out_float = in_gps[0] + in_gps[1]/60 + in_gps[2]/3600
    return out_float

def save_max_script(script_str: str, path: str='./resources/', app_string: str=''):
    with open(path + 'render_script' + app_string + '.ms', 'w') as f:
        f.write(script_str)

def get_exif_data(img_path: str) -> Tuple[datetime, Tuple[float, float], Tuple[int, int]]:
    with open(img_path, "rb") as img_file:
        exif_data = Image(img_file)
    
    if exif_data.has_exif:
        # Get capture time
        offset_time = float(exif_data.offset_time.replace(':','.'))
        dt = datetime.strptime(exif_data.datetime, '%Y:%m:%d %H:%M:%S')
        dt = dt + timedelta(hours=offset_time)

        # Get GPS data, if available
        try:
            gps_data = (exif_data.gps_latitude, exif_data.gps_longitude)
        except Exception as e:
            gps_data = [[55.0, 40.0, 33.38], [12.0, 33.0, 55.91]]
            print(''.join(['=' for i in range(50)]))
            print(e)
            print('Defaulting to CPH - DK')
            print("MAKE SURE TO SETUP DAYLIGHT SYSTEM MANUALLY IF NEEDED!")
            print(''.join(['=' for i in range(50)]))

        # Get image size
        img_size = [exif_data.pixel_x_dimension, exif_data.pixel_y_dimension]

    return dt, gps_data, img_size

def __prepare_directories(render_path, out_maxscript_path):
    # Script Path
    path = pathlib.Path(render_path + out_maxscript_path)
    path.mkdir(parents=True, exist_ok=True)

    # Sun Vector Path
    path = pathlib.Path(render_path + '/sun_vec/')
    path.mkdir(parents=True, exist_ok=True)

    # Cam Vector Path
    path = pathlib.Path(render_path + '/cam_vec/')
    path.mkdir(parents=True, exist_ok=True)

    # Sky Vis Path
    path = pathlib.Path(render_path + '/sky_vis/')
    path.mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(render_path + '/sky_vis/vray_raw')
    path.mkdir(parents=True, exist_ok=True)

def walp_generate_max_script(img_paths: Tuple[str], sensor_size: Tuple[float, float], render_path: str, out_maxscript_path: str='./maxscript/', out_maxscript_app_str: str='', camera_name: str='PhysCamera001', daylight_name: str='Daylight001', calib_path: str='./resources/', calib_app_str: str='', extrin_path: str='./resources/', extrin_app_str: str='ext_', single_cam_pos = False):
    __prepare_directories(render_path, out_maxscript_path)

    if isinstance(img_paths, str):
        img_paths = [img_paths]
    elif not isinstance(img_paths, Iterable):
        raise TypeError
    
    # Preproccess cam rvec and tvec for static camera
    rvec = np.zeros((3,1))
    tvec = np.zeros((3,1))
    if single_cam_pos:
        # get all marker estimates and take the mean
        files = glob.glob(extrin_path + '*.pkl')
        for file in files:
            t_rvec, t_tvec = marker_detection.load_extrinsics(file.split('.')[0], '')
            rvec = rvec + t_rvec
            tvec = tvec + t_tvec
        
        rvec = rvec / len(files)
        tvec = tvec / len(files)

    # =========================================================
    # Create empty string
    ms = ms_start()

    for img_path in img_paths:
        ms = ms + f'-- ===================================================\n'
        ms = ms + f'-- ============= NEW RENDER ==========================\n'
        ms = ms + f'-- ===================================================\n'
        ms = ms + f'\n'
        # Get data from image
        date_time, gps_data, img_size = get_exif_data(img_path)

        # Load camera intrinsic parameters
        mtx, dist = cam_calib_v2.load_calib(calib_path, calib_app_str)

        # Load camera extrinsic parameters
        if not single_cam_pos:
            rvec, tvec = marker_detection.load_extrinsics(extrin_path, extrin_app_str + img_path.split('\\')[-1].split('.')[0])

        # Generate render setup max script string
        raw_file_name = img_path.split('\\')[-1].split('.')[0]
        ms = ms + ms_render_setup(img_size, render_path, raw_file_name)
        ms = ms + ms_setup_daylight(daylight_name, date_time, gps_data)
        ms = ms + ms_cam_setup(mtx, tvec, rvec, camera_name, render_path, img_size, sensor_size)
        ms = ms + ms_render()
        ms = ms + ms_sun_vec(render_path)
        ms = ms + ms_cam_vec(render_path)

    # Add SkyVis render
    ms = ms + ms_sky_vis(['brick', 'table'], render_path)

    ms = ms + ms_end()

    save_max_script(ms, render_path + out_maxscript_path, out_maxscript_app_str)
            











if __name__ == "__main__":
    img_path = ['F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec\\DSC_3364.jpg', 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec\\DSC_3364.jpg']
    render_path = 'F:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiment_proff_test\\'

    sensor_size = [36,24]
    
    walp_generate_max_script(img_path, sensor_size, render_path)