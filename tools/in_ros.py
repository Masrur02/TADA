#!/usr/bin/env python3
from argparse import ArgumentParser
import rospy
import numpy as np
import time
import torch
import mmcv
import cv2
import torch.nn.functional as F
import matplotlib.cm as cm
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from sensor_msgs.msg import Image, CameraInfo
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis.inference import *
from tools.ros_utils import imgmsg_to_cv2, cv2_to_imgmsg
from message_filters import Subscriber, ApproximateTimeSynchronizer
import matplotlib
from dataset.common import colors_rugd, colors_city
from sensor_msgs.msg import PointCloud2
count = 0

# ---------------- Utility Functions ---------------- #

def update_legacy_cfg(cfg):
    cfg.data.test.pipeline[1]['img_scale'] = tuple(cfg.data.test.pipeline[1]['img_scale'])
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def visualize_segmap(seg_map, dataset):
    if dataset == 'rugd':
        seg_map[seg_map == 255] = 24
        colors_voc_origin = np.array(colors_rugd, dtype=np.uint8)
    else:
        seg_map[seg_map == 255] = 19
        colors_voc_origin = np.array(colors_city, dtype=np.uint8)
    new_im = colors_voc_origin[seg_map].astype(np.uint8)
    new_im = new_im[:, :, [2, 1, 0]]  # RGB → BGR
    return new_im,


def visualize_trav(trav, target_size=None):
    if torch.is_tensor(trav):
        trav = trav.detach().cpu().numpy()
    if trav.ndim == 4:
        trav = trav[0]
    if trav.ndim == 3 and trav.shape[0] == 1:
        trav = np.squeeze(trav, axis=0)
    score_map_norm = (trav - np.min(trav)) / (np.ptp(trav) + 1e-8)
    cmap = matplotlib.cm.get_cmap('jet')
    score_map_color = (cmap(score_map_norm)[:, :, :3] * 255).astype(np.uint8)
    if target_size is not None:
        score_map_color = cv2.resize(score_map_color, target_size, interpolation=cv2.INTER_LINEAR)
    return score_map_color


def parse_args():
    parser = ArgumentParser(description='IDANAV ROS deployment')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--dataset', default='city', help='Dataset type (city/rugd)')
    parser.add_argument('--opacity', type=float, default=0.5)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args



# ---------------- Main ROS Node ---------------- #
class IDANAV_Manager(object):
    def __init__(self, model, palette, opacity):
        self.model = model
        self.model.eval()
        self.pal = palette
        self.opacity = opacity
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.stdinv = 1 / np.float64(self.std.reshape(1, -1))

        

        # Publishers
        self.pub_rgb_proc = rospy.Publisher('rgb_proc', Image, queue_size=10)
        self.pub_seg2 = rospy.Publisher('seg_raw', Image, queue_size=10)
        self.pub_trav_visualize = rospy.Publisher('trav_vis', Image, queue_size=10)
        self.pub_trav_map = rospy.Publisher('trav_map', Image, queue_size=10)
        self.pub_trav_pov= rospy.Publisher('trav_map_pov', Image, queue_size=10)
       

        # Subscribers (RGB + Depth sync)
        self.rgb_sub = Subscriber('/D435i/color/image_raw', Image)
        self.depth_sub = Subscriber('/D435i/depth/color/points', PointCloud2)
        ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.sync_callback)
        print('Initialization finished.')

    
 

    def sync_callback(self, rgb_msg, depth_msg):
        global count
        count += 1
        if count % 5 != 0:
            return
        t1 = time.time()

        # Convert ROS → OpenCV
        rgb = imgmsg_to_cv2(rgb_msg).astype(np.float32)
        depth = depth_msg
        rgb = cv2.resize(rgb, (640, 480))
        #depth = cv2.resize(depth, (640, 480))

        # Normalize RGB for model
        cv2.subtract(rgb, self.mean, rgb)
        cv2.multiply(rgb, self.stdinv, rgb)
        model_input = rgb.transpose((2, 0, 1))
        model_input = torch.tensor(model_input).unsqueeze(0).to(torch.float32)
        img_meta = dict(ori_shape=rgb.shape[:2] + (3,), img_shape=rgb.shape[:2] + (3,),
                        pad_shape=rgb.shape[:2] + (3,), scale_factor=1.0, flip=False)

        with torch.inference_mode():
            result, trav = self.model(return_loss=False, img=[model_input], img_metas=[[img_meta]])
            seg_map = result[0]

        pred_img, = visualize_segmap(seg_map, args.dataset)
        pred_img = cv2.resize(pred_img, (640, 480))
        input_img = model_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
        input_img = (input_img * 255).astype(np.uint8)

        trav = F.interpolate(trav, size=model_input.shape[2:], mode="bilinear", align_corners=False)
        trav_np = trav.squeeze().detach().cpu().numpy().astype(np.float32)  
        
        trav_pov =(trav_np * 255).astype(np.uint8)
        
        
        trav_colormap = visualize_trav(trav)
        trav_colormap = cv2.cvtColor(trav_colormap, cv2.COLOR_RGB2BGR)
        trav_vis = cv2.addWeighted(input_img, 0.6, trav_colormap, 0.4, 0)
      

       

        msg_seg2 = cv2_to_imgmsg(pred_img, encoding='bgr8'); msg_seg2.header = rgb_msg.header
        msg_trav_raw = cv2_to_imgmsg(trav_np, encoding='32FC1');msg_trav_raw.header = rgb_msg.header
        msg_trav_pov= cv2_to_imgmsg(trav_pov, encoding='8UC1');msg_trav_raw.header = rgb_msg.header

        #msg_trav_vis = cv2_to_imgmsg(trav_vis, encoding='bgr8'); msg_trav_vis.header = rgb_msg.header
        msg_trav_vis = cv2_to_imgmsg(trav_vis, encoding='bgr8'); msg_trav_vis.header = rgb_msg.header
        msg_rgb_proc = cv2_to_imgmsg(input_img.astype(np.uint8), encoding='bgr8'); msg_rgb_proc.header = rgb_msg.header
        self.pub_rgb_proc.publish(msg_rgb_proc)
        self.pub_seg2.publish(msg_seg2)
        self.pub_trav_map.publish(msg_trav_raw)
        self.pub_trav_pov.publish(msg_trav_pov)
        self.pub_trav_visualize.publish(msg_trav_vis)

        print(f"FPS: {1.0 / (time.time() - t1):.2f}")
        del result, seg_map, trav
        torch.cuda.empty_cache()


# ---------------- Main ---------------- #
if __name__ == '__main__':
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test['type'] = 'MESHDataset'
    cfg.data.test['data_root'] = '/home/vail/Masrur/DA/MESH'

    cfg.model.test_cfg.mode = 'whole'
    cfg.model.test_cfg.batched_slide = False
    cfg.model.test_cfg.pop('crop_size', None)
    cfg.model.test_cfg.pop('stride', None)


    dataset = build_dataset(cfg.data.test)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu',
                                 revise_keys=[(r'^module\.', ''), ('model.', '')])
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.PALETTE = checkpoint.get('meta', {}).get('PALETTE', dataset.PALETTE)
    palette = model.PALETTE
    model = MMDataParallel(model, device_ids=[0])
    rospy.init_node('idanav_deploy', anonymous=True)
    node = IDANAV_Manager(model, palette, args.opacity)
    rospy.spin()
