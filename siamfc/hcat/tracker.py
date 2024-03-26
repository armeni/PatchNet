# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals
# import os
# from sys import path, exit
# env_path = os.path.join(os.path.dirname(__file__), '..')
# if env_path not in path:
#     path.append(env_path)
# from pysot_toolkit.config import backbone_path, model_path 
from argparse import ArgumentParser
import cv2

from got10k.trackers import Tracker as GTracker


import numpy as np
import math
import cv2
import torch
import torch.nn.functional as F
import time
import onnxruntime
import os
import glob

backbone_path = '/home/uavlab20/tracking/HCAT/backbone_res18_N2_q16.onnx'
model_path = '/home/uavlab20/tracking/HCAT/complete_res18_N2_q16.onnx'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TITLE = 'Tracking'


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    region = np.array(region)
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h

class Tracker(GTracker):

    def __init__(self, name, backbone_path, model_path, feature_size, window_penalty=0.49, penalty_k=0, exemplar_size=128, instance_size=256):
        self.name = name
        self.window_penalty = window_penalty
        self.penalty_k = penalty_k
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.feature_size = feature_size
        self.ort_session_z = onnxruntime.InferenceSession(backbone_path, providers=['CUDAExecutionProvider'])
        self.ort_session_x = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """

        resize_factor = original_sz / model_sz

        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1

        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        r, c, k = im.shape
        im_context = im[max(0,int(context_ymin)):min(int(context_ymax)+1,r),max(0,int(context_xmin)):min(int(context_xmax)+1,c),:]
        r_context, c_context, _ = im_context.shape
        r_context_resize = round(r_context / resize_factor)
        c_context_resize = round(c_context / resize_factor)
        left_pad = round(left_pad / resize_factor)
        top_pad = round(top_pad / resize_factor)
        right_pad = round(right_pad / resize_factor)
        bottom_pad = round(bottom_pad / resize_factor)
        im_patch_context = cv2.resize(im_context, (c_context_resize, r_context_resize))

        te_im = np.zeros([model_sz,model_sz,k])
        te_im[top_pad:top_pad + r_context_resize, left_pad:left_pad + c_context_resize, :] = im_patch_context

        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c_context_resize, :] = avg_chans
        if bottom_pad:
            te_im[r_context_resize + top_pad:, left_pad:left_pad + c_context_resize, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c_context_resize + left_pad:, :] = avg_chans


        # if not np.array_equal(model_sz, original_sz):
        #     im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = te_im
        _,r,c = im_patch.shape
        if not r == c == model_sz:
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def initialize(self, image, info: dict) -> dict:
        bbox = info['init_bbox']
        box = np.array([
            bbox[0] - 1 + (bbox[2] - 1) / 2,
            bbox[1] - 1 + (bbox[3] - 1) / 2,
            bbox[2], bbox[3]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]
        context = 0.5 * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * 255 / 127

        # print(box)
        # bbox = np.array([
        # box[1] - 1 + (box[3] - 1) / 2,
        # box[0] - 1 + (box[2] - 1) / 2,
        # box[3], box[2]], dtype=np.float32)
        # print(bbox)
        # input()

        tic = time.time()
        hanning = np.hanning(self.feature_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])
        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))
        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))
        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)
        # normalize
        self.mean_ = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
        self.std_ = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
        z_crop /= 255
        z_crop -= self.mean_
        z_crop /= self.std_
        # initialize template feature
        ort_inputs_z = {'zf': self.to_numpy(z_crop).astype(np.float32)}
        self.ort_outs = self.ort_session_z.run(None, ort_inputs_z)
        self.zf = self.ort_outs[:1]
        self.pos_template = self.ort_outs[1:]
        out = {'time': time.time() - tic}
        return out

    def trackHCAT(self, image, info: dict = None) -> dict:
        # calculate x crop size
        tic_all = time.time()
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        # get crop
        x_crop = self.get_subwindow(image, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)

        # normalize
        x_crop /= 255
        x_crop -= self.mean_
        x_crop /= self.std_

        ort_inputs_x = {'x': self.to_numpy(x_crop).astype(np.float32),
                        'feature_template':self.zf[-1].astype(np.float32),
                        'pos_template':self.pos_template[-1].astype(np.float32)}
        # tic = time.time()
        tic_model = time.time()
        outputs = self.ort_session_x.run(None,ort_inputs_x)
        model_time = time.time()-tic_model
        # print(time.time()-tic)
        score = self._convert_score(torch.Tensor(outputs[0]))
        pred_bbox = self._convert_bbox(torch.Tensor(outputs[1]))
        pscore = score
        # window penalty
        pscore = pscore * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        self.center = self.center_pos
        bbox = [int(cx - width / 2),
                int(cy - height / 2),
                int(width),
                int(height)
                ]
        
        all_time = time.time()-tic_all

        out = {'target_bbox': bbox,
               'best_score': pscore[best_idx],
               'model_time': model_time,
               'all_time': all_time}
        return out



class TrackerMain:
    def __init__(self):
        self.mouse_pressed = False
        self.gt_bbox = [None, None, None, None]
        self.initial = True

    def read_ground_truth_bbox(self, gt_bbox_file):
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

        
    def onMouse(self, event, x, y, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.initial = True
            self.mouse_pressed = True
            self.gt_bbox[:2] = [x, y]
            self.gt_bbox[2:] = [None, None]
        if event == cv2.EVENT_LBUTTONUP:
            if self.mouse_pressed:
                self.mouse_pressed = False
                self.gt_bbox[2:] = [x - self.gt_bbox[0], y - self.gt_bbox[1]]
                if abs(self.gt_bbox[2]) <= 10 or abs(self.gt_bbox[2]) <= 10:
                    self.gt_bbox = [None, None, None, None]
                else:
                    if self.gt_bbox[2] < 0:
                        self.gt_bbox[0], self.gt_bbox[2] = self.gt_bbox[0] + self.gt_bbox[2], -self.gt_bbox[2]
                    if self.gt_bbox[3] < 0:
                        self.gt_bbox[1], self.gt_bbox[3] = self.gt_bbox[1] + self.gt_bbox[3], -self.gt_bbox[3]
        if event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_pressed:
                self.gt_bbox[2:] = [x - self.gt_bbox[0], y - self.gt_bbox[1]]

    def main(self):
        tracker = Tracker(name='HCAT',
                          backbone_path=backbone_path,
                          model_path=model_path,
                          feature_size=16,
                          window_penalty=0.42, penalty_k=0,
                          exemplar_size=128, instance_size=256)
        
        sequence_folders = sorted(glob.glob('/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-train-2/sequences/*/'))
        if not sequence_folders:
            print("Error: No sequence folders found in the sequences directory.")
            return
        
        with open("/home/uavlab20/tracking/fps/hcat_VisDrone-train-2_fps_results_GPU.txt", 'w') as fps_file:
            for sequence_folder in sequence_folders:
                sequence_name = os.path.basename(sequence_folder.rstrip('/')) 
                output_file = "/home/uavlab20/tracking/HCAT/results_VisDrone-train-2_GPU/"f'{sequence_name}.txt'  
                image_files = sorted(glob.glob(os.path.join(sequence_folder, '*.jpg')))
                if not image_files:
                    print(f"Error: No image files found in the sequence folder {sequence_folder}.")
                    continue

                with open(output_file, 'w') as f:
                    self.gt_bbox = self.read_ground_truth_bbox(os.path.join('/home/uavlab20/tracking/Datasets/VisDrone2019-SOT-train-2/annotations/', f'{sequence_name}.txt'))
                    bbox = self.gt_bbox[0]
                    fps = []
                    for image_file, gt_bbox in zip(image_files, self.gt_bbox):
                        img = cv2.imread(image_file)
                        if img is None:
                            print(f"Error: Unable to read image file {image_file}")
                            continue

                        tic = cv2.getTickCount()

                        if gt_bbox is not None:
                            if self.initial:
                                init_info = {'init_bbox': bbox}
                                tracker.initialize(img, init_info)
                                self.initial = False
                            else:
                                outputs = tracker.trackHCAT(img)
                                bbox = list(map(int, outputs['target_bbox']))
                        
                        if bbox[0] != 'NaN':
                            cv2.rectangle(img, (bbox[0], bbox[1]),
                                        (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)
                        if gt_bbox[0] != 'NaN':
                            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                        (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 0, 0), 3)

                        # cv2.imshow(TITLE, img)

                        f.write(','.join(map(str, bbox)) + '\n')

                        key = cv2.waitKey(1)
                        if key == ord('q'):  # exit on 'q' or last image
                            cv2.destroyAllWindows()
                            exit()
                        toc = cv2.getTickCount() - tic
                        toc /= cv2.getTickFrequency()
                        fps.append(1/toc)
                        # print(f'fps: {1/toc}')
                self.initial = True
                avg_fps = sum(fps) / len(fps)
                fps_file.write(f'{sequence_name}, {avg_fps}\n')  

            cv2.destroyAllWindows()



if __name__ == '__main__':
    TrackerMain().main()
