#!/usr/bin/env python
# Helpful problem-specific info for landmark classification
# The script is used for labeling the image dataset.
# Kiri Wagstaff, 7/20/17

import matplotlib.image as mpimg

class_map = {0: 'other',
             1: 'crater',
             2: 'dark_dune',
             3: 'streak',
             4: 'bright_dune',
             5: 'impact',
             6: 'edge'}

# reverse_class_map needs to be consistent with class_map
reverse_class_map = {v: k for k,v in class_map.items()}

dataset = {} # dict from img to label
rel_img_path = 'map-proj/'
# open up the labeled data file
with open('labels-map-proj.txt') as labels:
  for line in labels:
    file_name, label = line.split(' ')
    img = mpimg.imread(rel_img_path + file_name)
    dataset[line] = int(label)
  

print(len(dataset))
