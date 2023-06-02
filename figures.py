import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

from PIL import Image
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataset_generation_const import DISTANCE_NORMALIZATION_FACTOR
from main import InferenceDataset
from model import SOCS
from segmentation_metrics import get_centroid_matches
from util import parse_train_step, get_checkpoint_path, MASK_COLORS

def overlay_masks_on_img(img_arr, mask_weights):
    """
    Given an input image (numpy array), superimpose colored masks
    """
    (H, W, _) = img_arr.shape
    img = Image.fromarray(img_arr)
    for (i, mask) in enumerate(mask_weights):
        color_arr = np.ones((H, W, 3)) * MASK_COLORS[i]
        color_img = Image.fromarray(color_arr.astype('uint8'))
        mask_img = Image.fromarray((mask * 255).astype('uint8'), mode='L')
        img = Image.composite(color_img, img, mask_img)
    return np.array(img)

def plot_seg_sequence(ckpt, batch, results, timepts, cam=1):
    """
    Show the input image sequence and predicted segmentation sequence over time.
    """
    (TC, H, W, _) = batch['img_seq'].shape
    if 'cameras' in ckpt.hparams and ckpt.hparams['cameras'] is not None:
        cameras = ckpt.hparams['cameras']
    else:
        cameras = [1]
    num_cameras = len(cameras)
    seq_len = ckpt.hparams['num_frame_slots'] // num_cameras
    img_dims = (TC, H, W)
    fig_width = 10
    fig_height_width_ratio = (img_dims[1] / img_dims[2]) * (2 / len(timepts))

    (fig, axes) = plt.subplots(2, len(timepts), figsize=fig_width*np.array([1, fig_height_width_ratio]))
    img_seq = (batch['img_seq'] * 255).astype('uint8')
    obj_weights = results['per_object_weights']
    for (i, t) in enumerate(timepts):
        frame_idx = np.ravel_multi_index((t, cam), (seq_len, num_cameras))
        axes[0, i].imshow(img_seq[frame_idx])
        axes[0, i].set_axis_off()
        seg = ckpt.show_object_masks(obj_weights, img_dims, idx=frame_idx)
        axes[1, i].imshow(seg)
        axes[1, i].set_axis_off()

    plt.tight_layout(pad=0, h_pad=0.5)
    return fig

def plot_seg_overlay(ckpt, batch, results, timepts, cam=1, chosen_inds=None):
    """
    Show the input image sequence with best-fit predicted masks superimposed on ground-truth 
    objects.
    """
    (TC, H, W, _) = batch['img_seq'].shape
    cameras = ckpt.hparams['cameras']
    C = len(cameras)
    T = TC // C
    obj_weights = torch.tensor(results['per_object_weights']).reshape((-1, T, C, H, W))
    gt_weights = torch.tensor(batch['instance_oh'].reshape((T, C, H, W, -1))).moveaxis(-1, 0)
    (_, pred_inds, gt_inds) = get_centroid_matches(obj_weights, gt_weights)
    chosen_pred_weights = obj_weights[pred_inds]
    if chosen_inds is not None:
        chosen_pred_weights = chosen_pred_weights[chosen_inds]

    fig_width = 10
    fig_height_width_ratio = (H / W) * (1 / len(timepts))
    (fig, axes) = plt.subplots(1, len(timepts), figsize=fig_width*np.array([1, fig_height_width_ratio]))
    img_seq = (batch['img_seq'] * 255).astype('uint8')
    
    for (i, t) in enumerate(timepts):
        frame_idx = np.ravel_multi_index((t, cam), (T, C))
        img = img_seq[frame_idx]
        img = overlay_masks_on_img(img, chosen_pred_weights[:, t, cam].numpy())
        axes[i].imshow(img)
        axes[i].set_axis_off()

    plt.tight_layout(pad=0, h_pad=0.5, w_pad=0.5)
    return fig

def plot_raw_seg_overlay(ckpt, batch, results, timepts, cam=1, chosen_inds=None):
    """
    Show the input image sequence with (selected) predicted masks superimposed.
    """
    (TC, H, W, _) = batch['img_seq'].shape
    cameras = ckpt.hparams['cameras']
    C = len(cameras)
    T = TC // C
    obj_weights = torch.tensor(results['per_object_weights']).reshape((-1, T, C, H, W))
    if chosen_inds is not None:
        obj_weights = obj_weights[chosen_inds]

    fig_width = 10
    fig_height_width_ratio = (H / W) * (1 / len(timepts))
    (fig, axes) = plt.subplots(1, len(timepts), figsize=fig_width*np.array([1, fig_height_width_ratio]))
    img_seq = (batch['img_seq'] * 255).astype('uint8')
    
    for (i, t) in enumerate(timepts):
        frame_idx = np.ravel_multi_index((t, cam), (T, C))
        img = img_seq[frame_idx]
        img = overlay_masks_on_img(img, obj_weights[:, t, cam].numpy())
        axes[i].imshow(img)
        axes[i].set_axis_off()

    plt.tight_layout(pad=0, h_pad=0.5, w_pad=0.5)
    return fig

# TODO remove OpenCV dependency (or add to requirements if it stays)
def plot_waypoints(ckpt, data_path, batch, results):
    """
    Plot the ground-truth and predicted future waypoints on the last image in the sequence.
    Note that this requires a special dataset containing full-res images and the ground-truth
    waypoints. Also requires OpenCV package.
    """
    import cv2
    with open(data_path, 'rb') as f:
        data = np.load(f)
        img_seq = data['full_rgb'] / 255
        loaded_intrinsics = data['intrinsics']
        loaded_extrinsics = data['extrinsics']
    img = img_seq[-1, 1]

    intrinsics_matrix = np.zeros((3, 3), dtype='double')
    intrinsics = loaded_intrinsics[1].flatten()
    intrinsics_matrix[0, 0] = intrinsics[0] # f_x
    intrinsics_matrix[0, 2] = intrinsics[2] # c_x
    intrinsics_matrix[1, 1] = intrinsics[1] # f_y
    intrinsics_matrix[1, 2] = intrinsics[3] # c_y
    intrinsics_matrix[2, 2] = 1
    intrinsics = torch.tensor(intrinsics_matrix, dtype=torch.double)
    extrinsics = torch.tensor(np.array([[0, -1,  0,  0],
                                        [0,  0, -1, loaded_extrinsics[0,1,0,3]],
                                        [1,  0,  0,  0],
                                        [0,  0,  0,  1]]), dtype=torch.double)

    img = img.copy()
    
    expert_waypoints = torch.tensor(batch['bc_waypoints'] * DISTANCE_NORMALIZATION_FACTOR, dtype=torch.double)
    expert_img_points = _get_img_points(intrinsics, extrinsics, expert_waypoints)
    for x, y in expert_img_points:
        cv2.circle(img, (x, y), 10, (0,1.,0), -1) # Green

    pred_waypoints = torch.tensor(results['bc_waypoints'].squeeze() * DISTANCE_NORMALIZATION_FACTOR, dtype=torch.double)
    pred_img_points = _get_img_points(intrinsics, extrinsics, pred_waypoints)
    for x, y in pred_img_points:
        cv2.circle(img, (x, y), 10, (.9,.6,0), -1) # Orange
    
    fig = plt.figure()
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.tight_layout()
    return fig

def _get_img_points(intrinsics, extrinsics, waypoints):
    """
    Find the (x, y) coordinates in the image plane for provided waypoints in space.
    """
    N = waypoints.shape[0]
    points = intrinsics.new_zeros((N, 3))
    points[:, :2] = waypoints

    # [num_points, 4]
    homogeneous_points = torch.nn.functional.pad(points, (0, 1), value=1)
    # [3, 4] @ [4, num_points] = [3, num_points]
    camera_points = extrinsics[:3, :] @ homogeneous_points.T
    # [3, 3] @ [3, num_points] = [3, num_points]
    homogeneous_image_points = intrinsics @ camera_points
    # Filter out points behind the camera origin.
    homogeneous_image_points = homogeneous_image_points[:, homogeneous_image_points[2] > 0]
    # [2, num_points]
    image_points = homogeneous_image_points[:2] / homogeneous_image_points[2]
    # [num_points, 2]
    int_image_points = image_points.T.round().int().numpy()
    return int_image_points
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_root')
    parser.add_argument('--out_path', default='figure.png')
    parser.add_argument('--cache_save', default=None)
    parser.add_argument('--cache_load', default=None)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--camera', type=int, choices=[0,1,2], help='Which camera to plot')
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--fig_type', choices=['seg_seq', 'seg_overlay', 'raw_seg_overlay', 'waypoints'], default='seg_seq')
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--parallel_pix', type=int, default=10000,
        help='Number of pixels to decode in each pass. More takes more memory but requires less passes as a result.')
    args = parser.parse_args()

    if args.log_root.endswith('.ckpt'):
        checkpoint_path = args.log_root
        checkpoint_fname = os.path.basename(checkpoint_path)
        log_dir = os.path.dirname(checkpoint_path)
        train_step = parse_train_step(checkpoint_fname)
    else:
        checkpoint_dir = os.path.join(args.log_root, 'checkpoints')
        (checkpoint_path, train_step) = get_checkpoint_path(checkpoint_dir)
        log_dir = args.log_root

    print(f'Loading checkpoint: {checkpoint_path}')
    ckpt = SOCS.load_from_checkpoint(checkpoint_path)

    if args.data_root is None:
        data_root = ckpt.hparams['dataset_name']
    else:
        data_root = args.data_root

    if args.split == 'train':
        add_instance_seg = False
        num_seq = 40000
        if args.data_root is None:
            data_root = os.path.join(data_root, 'train')
    else:
        add_instance_seg = True
        num_seq = 208
        if args.data_root is None:
            data_root = os.path.join(data_root, 'val')

    if args.cache_load:
        with open(args.cache_load, 'rb') as f:
            data = pickle.load(f)
            batch = data['batch']
            batch_results = data['batch_results']
    else:
        ckpt.inference_parallel_pixels = args.parallel_pix
        dataset = InferenceDataset(ckpt.hparams['sequence_len'],
                                ckpt.hparams['spatial_patch_hw'],
                                data_root=data_root,
                                num_sequences=num_seq,
                                img_dim_hw=ckpt.hparams['img_dim_hw'],
                                camera_choice=ckpt.hparams['cameras'],
                                decode_pixel_downsample_factor=1,
                                add_instance_seg=add_instance_seg,
                                no_viewpoint = not ckpt.hparams['provide_viewpoint'])
        dataset.set_indices([args.idx])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        if args.gpu is not None:
            trainer = Trainer(accelerator='gpu', devices=[args.gpu], logger=False)
        else:
            trainer = Trainer(accelerator='cpu', logger=False)

        batch = dataset.__getitem__(0)
        batch_results = trainer.predict(ckpt, dataloaders=dataloader)[0]

        if args.cache_save:
            full_results = dict(batch=batch, batch_results=batch_results)
            with open(args.cache_save, 'wb') as f:
                pickle.dump(full_results, f)

    frames = range(ckpt.hparams['sequence_len'])
    cam = args.camera
    if args.fig_type == 'seq_seq':
        fig = plot_seg_sequence(ckpt, batch, batch_results, frames, cam=cam)
    elif args.fig_type == 'seg_overlay':
        fig = plot_seg_overlay(ckpt, batch, batch_results, frames, cam=cam)
    elif args.fig_type == 'raw_seg_overlay':
        fig = plot_raw_seg_overlay(ckpt, batch, batch_results, frames, cam=cam) 
    elif args.fig_type == 'waypoints':
        data_path = os.path.join(data_root, f'{args.idx}.npz')
        fig = plot_waypoints(ckpt, data_path, batch, batch_results)
    fig.savefig(args.out_path)