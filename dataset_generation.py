import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

import argparse
import os
import numpy as np
import pickle

from PIL import Image
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import camera_segmentation_utils

from dataset_generation_const import *

CAM_TYPE = {open_dataset.CameraName.FRONT_LEFT: 0, open_dataset.CameraName.FRONT: 1, open_dataset.CameraName.FRONT_RIGHT: 2}

def save_train_seq(seq, bc_seq, seq_num, out_dir):
    num_cam = len(CAM_TYPE)
    temporal_transform = np.zeros((len(seq), 4, 4))
    timestamps = np.zeros((len(seq), num_cam))
    top_crop_ratio = 60/140
    top_offset = RESIZE_DIM[1]*top_crop_ratio 
    crop_ltrb = (0, top_offset, RESIZE_DIM[0], RESIZE_DIM[1])
    final_dim_hw = (RESIZE_DIM[1] - int(top_offset), RESIZE_DIM[0])
    rgbs = np.zeros((len(seq), num_cam, final_dim_hw[0], final_dim_hw[1], 3))
    extrinsics = np.zeros((len(seq), num_cam, 4, 4))
    
    for (t, frame) in enumerate(seq):
        temporal_transform[t] = np.array(frame.pose.transform).reshape((4,4))
        for image in frame.images:
            if image.name in CAM_TYPE:
                f = CAM_TYPE[image.name]
                timestamps[t, f] = frame.timestamp_micros
                rgb = tf.image.decode_jpeg(image.image).numpy()
                rgb_img = Image.fromarray(rgb).resize((RESIZE_DIM)).crop((crop_ltrb))
                rgbs[t, f] = np.array(rgb_img).astype('float') / 255
                
        for calibration in frame.context.camera_calibrations:
            if calibration.name in CAM_TYPE:
                f = CAM_TYPE[calibration.name]
                # sensor to vehicle frame
                extrinsics[t, f] = np.array(calibration.extrinsic.transform).reshape((4, 4))
                
    origin = np.expand_dims(temporal_transform[0], 0) # 1 x 4 x 4
    viewpoint_transform = np.expand_dims(np.linalg.solve(origin, temporal_transform), 1) @ extrinsics # T x C x 4 x 4
    viewpoint_transform[:, :, :3, 3] /= DISTANCE_NORMALIZATION_FACTOR
    viewpoint_transform = viewpoint_transform[:, :, :-1].reshape((len(seq), num_cam, -1))
    
    timestamps -= timestamps[0]
    timestamps /= TIMESTAMP_NORMALIZATION_FACTOR
    
    bc_waypoints = np.zeros((len(bc_seq), 2))
    bc_mask = np.zeros((len(bc_seq)))
    bc_origin = temporal_transform[-1] # 4 x 4
    for (t, frame) in enumerate(bc_seq):
        if frame is not None:
            bc_mask[t] = 1
            bc_transform = np.linalg.solve(bc_origin, np.array(frame.pose.transform).reshape((4,4)))
            bc_waypoints[t] = bc_transform[:2, 3].T / DISTANCE_NORMALIZATION_FACTOR # 1 x 2                 
      
    with open(os.path.join(out_dir, f'{seq_num}.npz'), 'wb') as f:
        np.savez_compressed(f, rgb=rgbs,
                            viewpoint_transform=viewpoint_transform,
                            time=timestamps,
                            bc_waypoints=bc_waypoints,
                            bc_mask=bc_mask)
        
    return 

def make_train_seqs(first_seq_num, unique_start_ids, in_dir, out_dir):
    seq_num = first_seq_num
    train_files = os.path.join(in_dir, '*.tfrecord')
    filenames = tf.io.matching_files(train_files)
    dataset = tf.data.TFRecordDataset(filenames)
    current_seq = []
    current_BC = [None for _ in range(BC_LEN)]
    current_t = 0
    new_seq = True
    collect_BC = False
    for data in tqdm(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        new_t = frame.timestamp_micros

        # Collecting image frames for autoencoding
        if new_seq or (not collect_BC and ((new_t - current_t > STRIDE*1e5 - 0.05e6) and (new_t - current_t < STRIDE*1e5 + 0.05e6))):
            if new_seq and new_t in unique_start_ids:
                continue
                
            current_seq.append(frame)
            new_seq = False
            current_t = new_t
            # Once we've collected enough image frames, switch to collecting trajectory info
            if len(current_seq) == SEQ_LEN:
                collect_BC = True
                seq_end_t = current_t
            
        # Collecting future trajectory information for BC
        elif collect_BC:
            elapsed_t = new_t - seq_end_t
            index = int(np.round(elapsed_t / (BC_STRIDE*1e5)) - 1)
            # Deal with timeskips
            if index < 0:
                index = np.inf
            if index < BC_LEN:
                current_BC[index] = frame
            if index >= BC_LEN - 1:
                save_train_seq(current_seq, current_BC, seq_num, out_dir)
                seq_start_t = current_seq[0].timestamp_micros
                unique_start_ids[seq_start_t] = seq_num
                seq_num += 1
                current_seq = []
                current_BC = [None for _ in range(BC_LEN)]
                new_seq = True
                collect_BC = False
            current_t = new_t
                
        # The frame we wanted with the correct timestamp is missing, so start over with a new sequence
        elif (new_t - current_t < 0) or (new_t - current_t > STRIDE*1e5 + 0.05e6):
            if new_t not in unique_start_ids:
                current_seq = [frame]
                current_t = new_t
                new_seq = False
            else:
                current_seq = []
                current_t = new_t
                new_seq = True

        # Note that if we don't meet either above condition, it just means we haven't reached the next frame with the
        # correct timestamp yet
    return seq_num

def save_val_seq(seq, seq_num, out_dir):
    panoptic_label_inds = range(len(seq))
    num_cam = len(CAM_TYPE)
    seg_protos = [0 for _ in range(num_cam*SEQ_LEN)]
    temporal_transform = np.zeros((len(seq), 4, 4))
    timestamps = np.zeros((len(seq), num_cam))
    top_crop_ratio = 60/140
    top_offset = RESIZE_DIM[1]*top_crop_ratio 
    crop_ltrb = (0, top_offset, RESIZE_DIM[0], RESIZE_DIM[1])
    final_dim_hw = (RESIZE_DIM[1] - int(top_offset), RESIZE_DIM[0])
    rgbs = np.zeros((len(seq), num_cam, final_dim_hw[0], final_dim_hw[1], 3))
    extrinsics = np.zeros((len(seq), num_cam, 4, 4))
    
    for (t, frame) in enumerate(seq):
        temporal_transform[t] = np.array(frame.pose.transform).reshape((4,4))
        for image in frame.images:
            if image.name in CAM_TYPE:
                f = CAM_TYPE[image.name]
                timestamps[t, f] = frame.timestamp_micros
                rgb = tf.image.decode_jpeg(image.image).numpy()
                rgb_img = Image.fromarray(rgb).resize((RESIZE_DIM)).crop((crop_ltrb))
                rgbs[t, f] = np.array(rgb_img).astype('float') / 255
                if t in panoptic_label_inds:
                    idx = np.ravel_multi_index((t, f), (SEQ_LEN, num_cam))
                    seg_protos[idx] = image.camera_segmentation_label
                
        for calibration in frame.context.camera_calibrations:
            if calibration.name in CAM_TYPE:
                f = CAM_TYPE[calibration.name]
                # sensor to vehicle frame
                extrinsics[t, f] = np.array(calibration.extrinsic.transform).reshape((4, 4))
                
    origin = np.expand_dims(temporal_transform[0], 0) # 1 x 4 x 4
    viewpoint_transform = np.expand_dims(np.linalg.solve(origin, temporal_transform), 1) @ extrinsics # T x C x 4 x 4
    viewpoint_transform[:, :, :3, 3] /= DISTANCE_NORMALIZATION_FACTOR
    viewpoint_transform = viewpoint_transform[:, :, :-1].reshape((len(seq), num_cam, -1))
    
    timestamps -= timestamps[0]
    timestamps /= TIMESTAMP_NORMALIZATION_FACTOR
    
    (panoptic_labels, _, panoptic_label_divisor) = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_protos(
        seg_protos, remap_values=True
    )
    semantic_segs = np.zeros(rgbs.shape[:-1], dtype='int')   
    instance_segs = np.zeros(rgbs.shape[:-1], dtype='int')
    for (i, label) in enumerate(panoptic_labels):
        (semantic_label_front, instance_label_front) = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
            label,
            panoptic_label_divisor)
        
        (t, f) = np.unravel_index(i, (SEQ_LEN, num_cam))
        semantic_label_img = Image.fromarray(semantic_label_front.astype('uint8').squeeze())
        semantic_label_img = semantic_label_img.resize(RESIZE_DIM, resample=Image.Resampling.NEAREST).crop(crop_ltrb)
        semantic_segs[t, f] = np.array(semantic_label_img).astype('int')
        instance_label_img = Image.fromarray(instance_label_front.astype('uint8').squeeze())
        instance_label_img = instance_label_img.resize(RESIZE_DIM, resample=Image.Resampling.NEAREST).crop(crop_ltrb)
        instance_segs[t, f] = np.array(instance_label_img).astype('int')
        
    with open(os.path.join(out_dir, f'{seq_num}.npz'), 'wb') as f:
        np.savez_compressed(f, rgb=rgbs,
                            semantic_seg=semantic_segs,
                            instance_seg=instance_segs,
                            viewpoint_transform=viewpoint_transform,
                            time=timestamps)
        
    return 

def collate_val_seqs(in_dir):
    """
    Find every sequence of frames with panoptic segmentation labels in the dataset.
    """
    val_files = os.path.join(in_dir, '*.tfrecord')
    filenames = tf.io.matching_files(val_files)
    dataset = tf.data.TFRecordDataset(filenames)
    seq_dict = {}
    for data in tqdm(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for image in frame.images:
            if image.name in CAM_TYPE:
                break
        if image.camera_segmentation_label.panoptic_label:
            seq_id = image.camera_segmentation_label.sequence_id
            if seq_id in seq_dict:
                seq_dict[seq_id].append(frame)
            else:
                seq_dict[seq_id] = [frame]

    return seq_dict

def make_val_seqs(out_dir, seq_dict):
    """
    Save frame sequences where there are no missing frames.
    """
    seq_num = 0
    for (_, seq) in tqdm(seq_dict.items()):
        seq = sorted(seq, key=lambda frame: frame.timestamp_micros)
        new_seq = True
        current_seq = []
        for frame in seq:
            new_t = frame.timestamp_micros
            if new_seq or ((new_t - current_t > STRIDE*1e5 - 0.05e6) and (new_t - current_t < STRIDE*1e5 + 0.05e6)):
                
                current_seq.append(frame)
                new_seq = False
                current_t = new_t
                if len(current_seq) == SEQ_LEN:
                    save_val_seq(current_seq, seq_num, out_dir)
                    seq_num += 1
                    current_seq = []
                    new_seq = True
            else:
                current_seq = [frame]
                current_t = new_t
                new_seq = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'val'])
    parser.add_argument('--in_dir', default='waymo_open_raw')
    parser.add_argument('--out_dir', default='waymo_open')
    parser.add_argument('--load_seq_ids', default=None,
        help='To resume generating training sequences, load previously generated IDs from file')
    parser.add_argument('--save_seq_ids', default=None,
        help='To resume generating training sequences later, save newly generated IDs to file')

    args = parser.parse_args()

    in_dir = os.path.join(args.in_dir, args.split)
    out_dir = os.path.join(args.out_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    if args.split == 'train':

        if args.load_seq_ids is not None:
            with open(args.load_seq_ids, 'rb') as f:
                unique_start_ids = pickle.load(f)
        else:
            unique_start_ids = {}

        prev_seq_num = len(unique_start_ids)
        new_seq_num = make_train_seqs(prev_seq_num, unique_start_ids, in_dir, out_dir)
        if args.save_seq_ids is not None:
            with open(args.save_seq_ids, 'wb') as f:
                pickle.dump(unique_start_ids, f)

        while new_seq_num > prev_seq_num and new_seq_num < MAX_NUM_TRAIN_SEQ:
            prev_seq_num = new_seq_num
            new_seq_num = make_train_seqs(prev_seq_num, unique_start_ids, in_dir, out_dir)
            
            if args.save_seq_ids is not None:
                with open(args.save_seq_ids, 'wb') as f:
                    pickle.dump(unique_start_ids, f)

    elif args.split == 'val':
        seq_dict = collate_val_seqs(in_dir)
        make_val_seqs(out_dir, seq_dict)