from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class FDST(data.Dataset):
  num_classes = 1
  default_resolution = [512, 512]
  mean = np.array([0.43149853,0.42709503,0.42243978],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.20863475,0.20143777,0.19956721],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(FDST, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'FDST')
    self.img_dir = os.path.join(self.data_dir, 'images')
    print("@@@@@@@@@@@@@@")
    print(self.img_dir)
    print("@@@@@@@@@@@@@@")
    if split == 'test':
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_{}2017.json').format(split)
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_extreme_{}2017.json').format(split)
      else:
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_{}2017.json').format(split)
  

    if split == 'val':
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'val_FDST.json')
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'train_FDST.json')
      if split == 'test':
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'test_FDST.json')
      else:
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'train_FDST.json')

          
    self.max_objs = 128
    self.class_name = ['__background__', 'person']
    self._valid_ids = [1]

    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing FDST 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = sorted(self.coco.getImgIds())
    print(self.images)
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = [[[] for __ in range(self.num_samples)] \
                  for _ in range(self.num_classes + 1)]
    for i in range(self.num_samples):
      img_id = self.images[i]
      for j in range(1, self.num_classes + 1):
        if isinstance(all_bboxes[img_id][j], np.ndarray):
          detections[j][i] = all_bboxes[img_id][j].tolist()
        else:
          detections[j][i] = all_bboxes[img_id][j]
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    os.system('python tools/reval.py ' + \
              '{}/results.json'.format(save_dir))
