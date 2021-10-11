import glob
from typing import List, Tuple
from collections.abc import Iterable
import numpy as np

import create_max_script
import misc
import cam_calib_v2
import marker_detection



def ms_unhide_valide_objs() -> str:
    ms = f'\n'
    
    ms = ms + f'obj = getNodeByName("valid_objs")\n'
    ms = ms + f'for o in obj.children do o.ishidden=false\n'

    return ms

def ms_hide_valide_objs() -> str:
    ms = f'\n'
    
    ms = ms + f'obj = getNodeByName("valid_objs")\n'
    ms = ms + f'for o in obj.children do o.ishidden=true\n'

    return ms

def ms_prep_render_settings() -> str:
    ms = f'\n'
    ms = ms + f'global r = renderers.current\n'

    ms = ms + f'r.gi_on = true\n'
    ms = ms + f'r.progressive_noise_threshold = 0.05\n'
    ms = ms + f'vfbControl #testresolution true \n'
    ms = ms + f'vfbControl #testresolution 2 \n'

    return ms

def ms_revert_render_settings() -> str:
    ms = f'\n'
    ms = ms + f'r.gi_on = false\n'
    ms = ms + f'r.progressive_noise_threshold = 0.01\n'
    ms = ms + f'vfbControl #testresolution false \n'
    ms = ms + f'vfbControl #testresolution 2 \n'

    return ms

def ms_enable_matte(matte_objcts_names: List[str]) -> str:
    ms = f'\n'
    
    for obj_name in matte_objcts_names:
        ms = ms + f'obj = getNodeByName("{obj_name}")\n'
        ms = ms + f'obj.material.matteSurface = true\n'
        ms = ms + f'obj.material.secondaryMatte = true\n'
        ms = ms + f'obj.material.matte_shadows = true\n'
        ms = ms + f'obj.material.matte_shadowsAffectAlpha = true\n'
        ms = ms + f'obj.material.alphaContribution = -1\n'

    ms = ms + f'\n'
    ms = ms + f'\n'

    return ms

def ms_disable_matte(matte_objcts_names: List[str]) -> str:
    ms = f'\n'
    
    for obj_name in matte_objcts_names:
        ms = ms + f'obj = getNodeByName("{obj_name}")\n'
        ms = ms + f'obj.material.matteSurface = false\n'

    ms = ms + f'\n'
    ms = ms + f'\n'

    return ms
    
def ms_enable_env() -> str:
    ms = f'\n'
    ms = ms + f'sceneMaterials["mat_env"].mapEnabled = #(true, false)\n'
    ms = ms + f'\n'

    return ms

def ms_disable_env() -> str:
    ms = f'\n'
    ms = ms + f'sceneMaterials["mat_env"].mapEnabled = #(false, true)\n'
    ms = ms + f'\n'

    return ms

def ms_prep_lighting(daylight_name: str) -> str:
    ms = f'\n'

    # get daylight system
    ms = f'd = getNodeByName("{daylight_name}")\n'

    # Change lighting to "sky vis lighting"
    ms = ms + f'd.sun.enabled = false\n'
    ms = ms + f'd.sky.on = false\n'
    
    # enable validation lights
    ms = ms + f'obj = getNodeByName("valid_light")\n'
    ms = ms + f'for o in obj.children do (\n'
    ms = ms + f'\to.ishidden=false\n'
    ms = ms + f'\to.on=true\n'
    ms = ms + f')\n'

    ms = ms + f'\n'
    ms = ms + f'\n'

    return ms

def ms_revert_lighting(daylight_name: str) -> str:
    ms = f'\n'

    # get daylight system
    ms = f'd = getNodeByName("{daylight_name}")\n'

    # Change lighting to "sky vis lighting"
    ms = ms + f'd.sun.enabled = true\n'
    ms = ms + f'd.sky.on = true\n'
    
    # Disable validation lights
    ms = ms + f'obj = getNodeByName("valid_light")\n'
    ms = ms + f'for o in obj.children do (e\n'
    ms = ms + f'\to.ishidden=true\n'
    ms = ms + f'\to.on=false\n'
    ms = ms + f')\n'

    ms = ms + f'\n'
    ms = ms + f'\n'

    return ms

def ms_render_setup(img_size) -> str:
    ms = f'\n'
    # render size
    #   .output_width : integer
    #   .output_height : integer
    ms = ms + f'r.output_width = {img_size[0]}\n'
    ms = ms + f'r.output_height = {img_size[1]}\n'
    ms = ms + f'renderWidth = {img_size[0]}\n'
    ms = ms + f'renderHeight = {img_size[1]}\n'
    ms = ms + f'\n'
    return ms

def ms_valid_light_setup(sun_rgb: Tuple[float, float, float], sky_rgb: Tuple[float, float, float], sun_pos: tuple[float, float, float]) -> str:
    sun_rgb = np.array(sun_rgb)
    sky_rgb = np.array(sky_rgb)
    # calc values
    sun_intensity = sum(sun_rgb) / len(sun_rgb)
    sun_color = 255 / max(sun_rgb) * sun_rgb
    
    sky_intensity = sum(sky_rgb) / len(sky_rgb)
    sky_color = 255 / max(sky_rgb) * sky_rgb

    ms = f'\n'

    # set color, intensity and position validation lights
    ms = ms + f'obj = getNodeByName("valid_light")\n'
    ms = ms + f'for o in obj.children do (\n'
    ms = ms + f'\tif o.name == "sun" then (\n'
    ms = ms + f'\t\to.color = color {sun_color[0]} {sun_color[1]} {sun_color[2]}\n'
    ms = ms + f'\t\to.multiplier = {sun_intensity}\n'
    ms = ms + f'\t\to.pos = [{sun_pos[0]*1000}, {sun_pos[1]*1000}, {sun_pos[2]*1000}]\n'
    ms = ms + f'\t)\n'
    ms = ms + f'\tif o.name == "dome" then (\n'
    ms = ms + f'\t\to.color = color {sky_color[0]} {sky_color[1]} {sky_color[2]}\n'
    ms = ms + f'\t\to.multiplier = {sky_intensity}\n'
    ms = ms + f'\t)\n'
    ms = ms + f')\n'

    return ms

def ms_env_map_setup(img_file):
    ms = f'\n'
    ms = ms + f'sceneMaterials["EnvMapSwitcher"].Background_map.HDRIMapName = "{img_file}"\n'
    ms = ms + f'sceneMaterials["EnvMapSwitcher"].Background_map.color_space = 0\n'
    ms = ms + f'sceneMaterials["EnvMapSwitcher"].Environment_map.HDRIMapName = "{img_file}"\n'
    ms = ms + f'sceneMaterials["EnvMapSwitcher"].Environment_map.color_space = 0\n'
    # sceneMaterials["EnvMapSwitcher"].Background_map.


    return ms

def create_valid_ms_render_script(matt_objcts_names: Tuple[str], img_paths: Tuple[str], sun_pos_file: str, render_path: str, data_file: str, extrin_path: str, sensor_size: Tuple[float, float]=[36,24], daylight_name='Daylight001', camera_name: str='PhysCamera001', extrin_app_str: str='ext_'):
    # TODO Make one for this script!
    # __prepare_directories(render_path, out_maxscript_path)

    # Load data
    data = misc.load_pckl_data(data_file)

    # Load Sun Pos
    sun_positions = misc.load_vector_from_file(sun_pos_file, normalise=False)

    if isinstance(img_paths, str):
        img_paths = [img_paths]
    elif not isinstance(img_paths, Iterable):
        raise TypeError

    ms = create_max_script.ms_start()

    # TODO order list
    #+ Setup Matte Material
    ms = ms + ms_enable_matte(matt_objcts_names)

    #+ Turn off Daylight System
    ms = ms + ms_prep_lighting(daylight_name)

    # Set render settings
        # GI on
        # Quality
        # Render Scale
    ms = ms + ms_prep_render_settings()

    # Unhide validation objects
    ms = ms + ms_unhide_valide_objs()

    # enable env map
    ms = ms + ms_enable_env()

    # Per Image
    for idx, (img_path, data_frame, sun_pos) in enumerate(zip(img_paths, data, sun_positions)):
        ms = ms + f'-- ===================================================\n'
        ms = ms + f'-- ============= NEW RENDER ==========================\n'
        ms = ms + f'-- ===================================================\n'
        ms = ms + f'\n'
        # Get data from image
        date_time, gps_data, img_size = create_max_script.get_exif_data(img_path)

        # Load camera intrinsic parameters
        mtx, dist = cam_calib_v2.load_calib()

        # Load camera extrinsic parameters
        rvec, tvec = marker_detection.load_extrinsics(extrin_path, extrin_app_str + img_path.split('\\')[-1].split('.')[0])

        # Setup per. image render settings
        file_name = img_path.split('\\')[-1].split('.')[0]
        ms = ms + create_max_script.ms_render_setup(img_size, render_path, file_name, output_raw=False, output_split=False)
        # ms = ms + ms_render_setup(img_size, render_path, file_name)

        # Set camera position
        ms = ms + create_max_script.ms_cam_setup(mtx, tvec, rvec, camera_name, render_path, img_size, sensor_size)

        # Set Lighting
        ms = ms + ms_valid_light_setup(data_frame[0], data_frame[1], sun_pos)

        # setup env maps
        ms = ms + ms_env_map_setup(img_path)

        # Render
        split_path = render_path.replace('\\', '/')
        ms = ms + create_max_script.ms_render(split_path + file_name + '.jpg', save_vfb_img=True)

    # Turn on Daylight System
    ms = ms + ms_revert_lighting(daylight_name)

    # Revert render settings
        # GI on
        # Quality
        # Render Scale
    ms = ms + ms_revert_render_settings()

    # Hide validation objects
    ms = ms + ms_hide_valide_objs()

    ms = ms + ms_disable_matte(matt_objcts_names)

    # disable env map
    ms = ms + ms_disable_env()

    ms = ms + create_max_script.ms_end()

    create_max_script.save_max_script(ms, render_path)

if __name__ == "__main__":
    folder_path = 'D:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\experiments\\experiemnt_0_pre_test_1\\1sec\\'
    img_folder_name = 'p\\'
    output_folder = '_output\\augmentations\\'
    data_file = '_output\\sun_sky_data_spec_danger'
    extrin_path = 'ext_res\\'
    sun_pos_file = 'sun_vec\\sun_vec.txt'

    # Get all images
    files = glob.glob(folder_path + img_folder_name + '*.jpg')

    create_valid_ms_render_script(['brick', 'table'], files, folder_path+sun_pos_file, folder_path+output_folder, folder_path+data_file, folder_path+extrin_path)
    pass