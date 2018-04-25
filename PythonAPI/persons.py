#!/usr/bin/python2

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import cv2

# data location
dataDir = '/media/tano/Elements/datasets/coco/'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
outputDir = "/tmp/coco_" + dataType
verbose = False

# create ouput directories
try:
  if os.path.exists(outputDir):
    shutil.rmtree(outputDir)
  print("Creating output dir")
  os.makedirs(outputDir)
  os.makedirs(outputDir + "/img")
  os.makedirs(outputDir + "/masks_machine")
except:
  print("Cannot create output dir")

# initialize COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing persons
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
# imgIds = coco.getImgIds(imgIds=[324158])
person_image_lst = coco.loadImgs(imgIds)

# save all images and masks to output dir
for image in person_image_lst:
  # get the cv image
  cvim = io.imread(image['coco_url'])
  if cvim is not None:
    print("Downloaded", image['coco_url'])
  if verbose:
    plt.imshow(cvim)
    plt.axis('off')

  # load and display instance annotations
  annIds = coco.getAnnIds(imgIds=image['id'], catIds=catIds, iscrowd=None)
  anns = coco.loadAnns(annIds)
  if verbose:
    coco.showAnns(anns)
    plt.show()

  # convert to semantic mask
  mask = coco.annToMask(anns[0])
  for ann in anns:
    # add to mask
    mask = np.logical_or(coco.annToMask(ann), mask)

  # show mask
  if verbose:
    plt.imshow(mask)
    plt.show()

  # transpose image for BGR
  if len(cvim.shape) == 3:
    cvim = cv2.cvtColor(cvim, cv2.COLOR_RGB2BGR)
  elif len(cvim.shape) == 2:
    cvim = cv2.cvtColor(cvim, cv2.COLOR_GRAY2BGR)

  # save in log folder
  cv2.imwrite(outputDir + "/img/" +
              str(image['id']) + ".png", cvim.astype(np.uint8))
  cv2.imwrite(outputDir + "/masks_machine/" +
              str(image['id']) + ".png", mask.astype(np.uint8))
