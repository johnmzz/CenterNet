from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import math

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  # Load count file
  count_dict = {}
  with open("../data/FDST/count.txt") as count_file:
    counts = count_file.readlines()
    for count in counts:
      name = count.split(':')[0]
      number = count.split(':')[1].rstrip()
      count_dict[name] = number

  # CC evaluation list
  detection_nums = {}
  detection_errors = {}
  print("@@@@@ count dict @@@@@")
  print(count_dict)

  # if video or webcam
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:          # find last index of '.', all letter right of it (video format), check if name is in the list video_ext
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)     ###### detector ######
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):     # if given entire directory
      print ("HHHHHHHHHHHHHHHHHH")
      image_names = []
      ls = os.listdir(opt.demo)     # return list of all file names in the directory
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))     # if image extension satisfied, add to "image_names" list
    else:
      image_names = [opt.demo]      # if single image, only that image name added to "image_names" list
    
    for (image_name) in image_names:
      ret = detector.run(image_name)      ###### detector ######
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      #print(time_str)

      img_name = image_name.split('/')[-1]
      print("!!!!!")
      print(ret["num_people"])
      detection_nums[img_name] = ret["num_people"]
    print(detection_nums)

    # Calculate MAE
    num_1 = len(detection_nums)
    num_2 = len(count_dict)
    print(num_1)
    print(num_2)

    MAE = 0
    MSE = 0
    for filename, count_num in detection_nums.items():
        filename = filename.replace(".png",".txt")
        filename = filename.replace(".jpg",".txt")
        print(filename + " --- " + "predicted: " + str(count_num) + ", " + "ground truth: " + count_dict[filename] + "\n")
        MAE += abs(int(count_num) - int(count_dict[filename]))
        MSE += (int(count_num) - int(count_dict[filename]))**2
    
    MAE = MAE / num_1
    MSE = math.sqrt(MSE / num_1)
    print(MAE)
    print(MSE)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
