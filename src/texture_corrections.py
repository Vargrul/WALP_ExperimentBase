import glob
import pathlib
import cv2 as cv
from tqdm import tqdm
import numpy as np

def main(path_folder: str, multiplier: float):
    # get all images in folder
    files = glob.glob(path_folder + '*.jpg')

    # Backup Images in /bu folder
    path = pathlib.Path(path_folder + 'bu\\')
    path.mkdir(parents=True, exist_ok=True)

    # for file in tqdm(files, ncols=100, desc='Backing up images'):
    #     img = cv.imread(file)
    #     cv.imwrite(path_folder + 'bu\\' + file.split('\\')[-1], img)

    for file in tqdm(files, ncols=100, desc='Correcting Images'):
        img = cv.imread(file)
        # correct images
        b = img[:,:,0] * multiplier[0]
        g = img[:,:,1] * multiplier[1]
        r = img[:,:,2] * multiplier[2]

        img = np.stack((b,g,r), axis=2)

        # Overwrite Images
        # cv.imshow('', img)
        # cv.waitKey(0)
        cv.imwrite(file, img)


if __name__ == "__main__":
    folder = 'D:\\OneDrive - Aalborg Universitet\\Projects\\World as a Light Probe\\data\\models\\ErimittageSlottet\\'
    # BRG
    multiplier = [1.0738255033557046979865771812081, 1.1111111111111111111111111111, 1.1940298507462686567164179104478]

    main(folder, multiplier)