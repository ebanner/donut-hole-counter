#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml

with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

project_dir = "/Users/edward/Downloads/project-1-at-2025-11-03-21-25-1dab8c86"

data.keys()


# # Strip EXIF data

# In[19]:


from pathlib import Path
from PIL import Image, ImageOps

img_basename = f'{project_dir}/images'

for img_fname in data['convert']:
    img_path = f'{img_basename}/{img_fname}'
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    img.save(img_path)


# # Copy Train Examples

# In[2]:


from tqdm import tqdm

for stem in tqdm(data['train']):
    get_ipython().system('cp {project_dir}/images/{stem}.jpeg train/images')
    get_ipython().system('cp {project_dir}/labels/{stem}.txt train/labels')


# # Copy Val Examples

# In[3]:


from tqdm import tqdm

for stem in tqdm(data['val']):
    get_ipython().system('cp {project_dir}/images/{stem}.jpeg val/images')
    get_ipython().system('cp {project_dir}/labels/{stem}.txt val/labels')


# # Copy Test Examples

# In[4]:


from tqdm import tqdm

for stem in tqdm(data['test']):
    get_ipython().system('cp {project_dir}/images/{stem}.jpeg test/images')
    get_ipython().system('cp {project_dir}/labels/{stem}.txt test/labels')

