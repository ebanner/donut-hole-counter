# run with ipython

from pathlib import Path

import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

with open("data.yaml", "r") as f:
    DATA = yaml.safe_load(f)

PROJECT_DIR = "/Users/edward/Downloads/project-1-at-2025-11-05-21-30-1dab8c86"


if __name__ == '__main__':
    #
    # Delete existing data
    #
    get_ipython().system('rm -f {train,val,test}/images/*')
    get_ipython().system('rm -f {train,val,test}/labels/*')

    #
    # Strip EXIF data
    #
    img_basename = f'{PROJECT_DIR}/images'
    for img_fname in DATA['convert']:
        img_path = f'{img_basename}/{img_fname}'
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img.save(img_path)


    #
    # Copy Train Examples
    #
    for stem in tqdm(DATA['train']):
        get_ipython().system('cp {PROJECT_DIR}/images/{stem}.jpeg train/images')
        get_ipython().system('cp {PROJECT_DIR}/labels/{stem}.txt train/labels')


    #
    # Copy Val Examples
    #
    for stem in tqdm(DATA['val']):
        get_ipython().system('cp {PROJECT_DIR}/images/{stem}.jpeg val/images')
        get_ipython().system('cp {PROJECT_DIR}/labels/{stem}.txt val/labels')


    #
    # Copy Test Examples
    #
    for stem in tqdm(DATA['test']):
        get_ipython().system('cp {PROJECT_DIR}/images/{stem}.jpeg test/images')
        get_ipython().system('cp {PROJECT_DIR}/labels/{stem}.txt test/labels')

    #
    # Delete weird image
    #
    get_ipython().system('rm -f val/images/4b03b9ad-IMG_1353.jpeg')
    get_ipython().system('rm -f val/labels/4b03b9ad-IMG_1353.txt')
