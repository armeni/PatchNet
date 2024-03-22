from __future__ import absolute_import

import os
import sys
import torch
sys.path.append('.')
import glob
import numpy as np

from siamfc import VariablePatchSiam
from patchnet import PatchNet


if __name__ == '__main__':
    seq_dir = os.path.expanduser('../Datasets/UAV123/sequences/uav1')
    img_files = sorted(glob.glob(seq_dir + '/*.jpg'))
    # try:
    anno = np.loadtxt('../Datasets/UAV123/anno/UAV123/uav1.txt', delimiter= ',')
    
    # except:
    #     anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    patchnet_path = 'pretrained/patchnet.pth'
    patchnet = PatchNet()
    patchnet.create_architecture()
    patchnet.load_state_dict(torch.load(patchnet_path)) # map_location=torch.device('cpu') for CPU
    patchnet.eval()
    patchnet.cuda()
    tracker = VariablePatchSiam(net_path, patchnet, interval=10) # 0 for only HCAT
    tracker.track(img_files, anno[0], visualize=True)
