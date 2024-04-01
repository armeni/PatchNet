from __future__ import absolute_import

import os
import sys
import torch
sys.path.append('.')
import glob
import numpy as np
import cv2

from siamfc import VariablePatchSiam
from patchnet import PatchNet

def read_ground_truth_bbox(gt_bbox_file):
        if os.path.exists(gt_bbox_file):
            with open(gt_bbox_file, 'r') as f:
                lines = f.readlines()
                bboxes = []
                for line in lines:
                    if 'NaN' not in line:
                        bbox = list(map(int, line.strip().split(',')))
                        bboxes.append(bbox)
                    else:
                        bboxes.append(['NaN','NaN','NaN','NaN'])
                return bboxes
        else:
            print(f"Error: Ground truth bounding box file not found for {gt_bbox_file}")
            return []
        

if __name__ == '__main__':
#     sequence_folders = sorted(glob.glob('/home/uavlab20/tracking/Datasets/UAV123/sequences/*/'))
#     with open("/home/uavlab20/tracking/fps/PatchNet_HCAT_5000_uav_fps_results_GPU.txt", 'w') as fps_file:
#         for sequence_folder in sequence_folders:
#             sequence_name = os.path.basename(sequence_folder.rstrip('/')) 
#             output_file = "/home/uavlab20/tracking/PatchNet/results_hcat_5000_uav_GPU/"f'{sequence_name}.txt'  
#             image_files = sorted(glob.glob(os.path.join(sequence_folder, '*.jpg')))
#             if not image_files:
#                 print(f"Error: No image files found in the sequence folder {sequence_folder}.")
#                 continue

#             with open(output_file, 'w') as f:
#                 gt_bbox = read_ground_truth_bbox(os.path.join('/home/uavlab20/tracking/Datasets/UAV123/anno/UAV123', f'{sequence_name}.txt'))
#                 bbox = gt_bbox[0]
#                 net_path = 'pretrained/siamfc_alexnet_e50.pth'
#                 patchnet_path = 'pretrained/patchnet.pth'
#                 patchnet = PatchNet()
#                 patchnet.create_architecture()
#                 patchnet.load_state_dict(torch.load(patchnet_path)) # map_location=torch.device('cpu') for CPU
#                 patchnet.eval()
#                 patchnet.cuda()
#                 tracker = VariablePatchSiam(net_path, patchnet, interval=5000) # 0 for only HCAT
#                 bbox, avg_fps = tracker.track(image_files, gt_bbox[0], visualize=True)
#                 for i in bbox:
#                     f.write(','.join(map(str, i)) + '\n')
#                 fps_file.write(f'{sequence_name}, {avg_fps}\n')  



    # for video
    gt_bbox = [None, None, None, None]
    cap = cv2.VideoCapture('/home/uavlab20/Downloads/car.mp4')
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    patchnet_path = 'pretrained/patchnet.pth'
    patchnet = PatchNet()
    patchnet.create_architecture()
    patchnet.load_state_dict(torch.load(patchnet_path)) # map_location=torch.device('cpu') for CPU
    patchnet.eval()
    patchnet.cuda()
    tracker = VariablePatchSiam(net_path, patchnet, interval=7) # 0 for only HCAT
    bbox, avg_fps = tracker.track(cap, gt_bbox, visualize=True)