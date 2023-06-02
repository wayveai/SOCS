import argparse
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import yaml

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import InferenceDataset
from model import SOCS
from util import parse_train_step, get_checkpoint_path

PLOT_CHOICES = ['ground_truth_rgb', 
                'ground_truth_seg', 
                'greedy_pred_rgb',
                'mixture_pred_rgb',
                'pred_seg',
                'pred_seg_foreground',
                'pixel_score']
PLOT_CHOICES = { key: idx for (idx, key) in enumerate(PLOT_CHOICES) }

def render_fig(fig):
    """Renders a figure into an RGB image."""
    canvas = fig.canvas
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    w, h = canvas.get_width_height()
    return image.reshape([h, w, 3])

def get_hparams(logdir):
    with open(os.path.join(logdir, 'hparams.yaml'), 'r') as f:
        hparams = yaml.safe_load(f)
    return hparams

def plot_frame_sequence_from_single_batch(ckpt, batch, results, plot_types, fig_width=10):
    img_dims = tuple(batch['decode_dims'])
    num_rows = len(plot_types)
    if ckpt.hparams['cameras'] is not None:
        cameras = ckpt.hparams['cameras']
    else:
        cameras = [1]
    num_cameras = len(cameras)
    seq_len = ckpt.hparams['sequence_len']

    fig_imgs = []
    img_seq_dims = (seq_len, num_cameras, img_dims[1], img_dims[2])
    img_seq = (batch['img_seq'] * 255).astype('uint8').reshape(img_seq_dims + (3,))
    obj_weights = results['per_object_weights']
    fig_height_width_ratio = (img_dims[1] / img_dims[2]) * (num_rows / num_cameras)

    for frame in range(seq_len):
        (f, axes) = plt.subplots(num_rows, num_cameras, figsize=fig_width*np.array([1, fig_height_width_ratio]))
        # Make sure axes have 2 dims even in case where only 1 row and/or camera
        axes = axes.reshape(num_rows, num_cameras)
        for (row, plot_type) in enumerate(plot_types):
            for cam in range(num_cameras):
                frame_idx = np.ravel_multi_index((frame, cam), (seq_len, num_cameras))
                if plot_type == PLOT_CHOICES['ground_truth_rgb']:
                    im = img_seq[frame, cam]
                elif plot_type == PLOT_CHOICES['ground_truth_seg']:
                    im = ckpt.show_ground_truth_seg(batch['instance_oh'], img_dims, idx=frame_idx)
                elif plot_type == PLOT_CHOICES['mixture_pred_rgb']:
                    im = ckpt.reconstruct_image(results['preds'], img_dims, idx=frame_idx)
                elif plot_type == PLOT_CHOICES['greedy_pred_rgb']:
                    im = ckpt.reconstruct_image(results['greedy_preds'], img_dims, idx=frame_idx)     
                elif plot_type == PLOT_CHOICES['pred_seg']:
                    im = ckpt.show_object_masks(obj_weights, img_dims, idx=frame_idx)
                elif plot_type == PLOT_CHOICES['pred_seg_foreground']:
                    foreground_seg = batch['instance_mask'].reshape(img_seq_dims)
                    im = ckpt.show_object_masks_foreground(obj_weights, foreground_seg, img_dims, idx=frame_idx)
                elif plot_type == PLOT_CHOICES['pixel_score']:
                    im = ckpt.show_pixel_scores(obj_weights, batch['instance_oh'], img_dims, idx=frame_idx)
                axes[row, cam].imshow(im)
                axes[row, cam].set_axis_off()

        plt.tight_layout(pad=0, h_pad=0.5)
        fig_imgs.append(render_fig(f))
        plt.close(f)

    return fig_imgs

def plot_seg_sequence(ckpt, batch, results, timepts, cam=1):
    (TC, H, W, _) = batch['img_seq'].shape
    if 'cameras' in ckpt.hparams and ckpt.hparams['cameras'] is not None:
        cameras = ckpt.hparams['cameras']
    else:
        cameras = [1]
    num_cameras = len(cameras)
    seq_len = ckpt.hparams['sequence_len']
    img_dims = (seq_len, num_cameras, H, W)

    (fig, axes) = plt.subplots(2, len(timepts))
    img_seq = (batch['img_seq'] * 255).astype('uint8')
    obj_weights = results['per_object_weights']
    for t in range(timepts):
        frame_idx = np.ravel_multi_index((t, cam), (seq_len, num_cameras))
        axes[0, t].imshow(img_seq[0, t, cam])
        seg = ckpt.show_object_masks(obj_weights, img_dims, idx=frame_idx)
        axes[1, t].imshow(seg)

    plt.tight_layout(pad=0, h_pad=0.5)
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_root', help='Path to log directory or specific checkpoint')
    parser.add_argument('--name', default=None)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--split', default='both', choices=['train', 'val', 'both'])
    parser.add_argument('--idx', type=int, default=None, nargs='+')
    parser.add_argument('--idx_file', default=None)
    parser.add_argument('--num_seq_to_analyze', type=int, default=1)
    parser.add_argument('--num_seq_to_plot', type=int, default=1)
    parser.add_argument('--num_train_seq', type=int, default=40000)
    parser.add_argument('--num_val_seq', type=int, default=208)
    parser.add_argument('--video_format', default='both', choices=['gif', 'mp4', 'both'])
    parser.add_argument('--gpu', type=int, default=None, nargs='+')
    parser.add_argument('--parallel_pix', type=int, default=10000,
        help='Number of pixels to decode in each pass. More takes more memory but requires less passes as a result.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--plot_types', 
                        default=['ground_truth_rgb', 'mixture_pred_rgb', 'pred_seg'], 
                        nargs='+', choices=PLOT_CHOICES.keys())
    
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
    ckpt.inference_parallel_pixels = args.parallel_pix

    if args.idx is not None:
        train_indices = args.idx
        val_indices = args.idx
    elif args.idx_file is not None:
        with open(args.idx_file, 'r') as f:
            indices = yaml.safe_load(f)
            train_indices = indices['train']
            val_indices = indices['val']
            if train_indices is None:
                train_indices = []
            if val_indices is None:
                val_indices = []
            if args.name is None:
                args.name = os.path.basename(args.idx_file).split('.')[0]
    else:
        train_indices = np.random.choice(range(args.num_train_seq), args.num_seq_to_analyze, replace=False)
        val_indices = np.random.choice(range(args.num_val_seq), args.num_seq_to_analyze, replace=False)

    if (args.split == 'both' or args.split == 'train') and len(train_indices) > 0:
        do_train = True
    else:
        do_train = False
    if (args.split == 'both' or args.split == 'val') and len(val_indices) > 0:
        do_val = True
    else:
        do_val = False

    if args.data_root is None:
        data_root = ckpt.hparams['dataset_root']
    else:
        data_root = args.data_root
    
    img_dim_hw = ckpt.hparams['img_dim_hw']

    if 'cameras' in ckpt.hparams and ckpt.hparams['cameras'] is not None:
        cameras = ckpt.hparams['cameras']
    else:
        cameras = [1]

    print(ckpt.hparams)

    train_data_root = os.path.join(data_root, 'train')
    val_data_root = os.path.join(data_root, 'val')

    plot_types = [ PLOT_CHOICES[choice] for choice in args.plot_types ]

    result_categories = ['avg_centroid_dist',
                         'reconstruction_err', 
                         'instance_reconstruction_err', 
                         'ari',
                         'seq_ari']
    
    def run_analysis(split, num_seq, indices, data_root):
        add_instance_seg = True if split == 'val' else False
        dataset = InferenceDataset(ckpt.hparams['sequence_len'],
                                   ckpt.hparams['spatial_patch_hw'],
                                   data_root=data_root,
                                   num_sequences=num_seq,
                                   img_dim_hw=img_dim_hw,
                                   camera_choice=cameras,
                                   decode_pixel_downsample_factor=1,
                                   add_instance_seg=add_instance_seg,
                                   no_viewpoint = not ckpt.hparams['provide_viewpoint'])
        dataset.set_indices(indices)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        if args.gpu is not None:
            trainer = Trainer(gpus=args.gpu, strategy="ddp" if len(args.gpu) > 1 else None, logger=False)
        else:
            trainer = Trainer(accelerator='cpu', logger=False)
        r = trainer.predict(ckpt, dataloaders=dataloader)

        results = {key: [] for key in result_categories}
        all_plots = []
        for (i, batch_results) in tqdm(enumerate(r)):
            batch = dataset.__getitem__(i)
            plots = plot_frame_sequence_from_single_batch(ckpt, batch, batch_results, plot_types)
            for key in result_categories:
                if key in batch_results:
                    results[key].append(batch_results[key])
            if i < args.num_seq_to_plot:
                all_plots.append(plots)

        print(f'Results on {split} dataset:')
        for (key, val) in results.items():
            print(f'Mean {key}: {np.nanmean(val)}, std: {np.nanstd(val)}')
        if args.num_seq_to_plot > 0:
            for (index, frames) in zip(indices, all_plots):
                if args.video_format == 'both' or args.video_format == 'gif':
                    imageio.mimwrite(os.path.join(log_dir, f'{split}_{index}_{train_step}.gif'), frames, fps=2)
                if args.video_format == 'both' or args.video_format == 'mp4':
                    imageio.mimwrite(os.path.join(log_dir, f'{split}_{index}_{train_step}.mp4'), frames, fps=2)
    
        return results
    
    overall_results = {'train_step': train_step}
    if do_train:
        results = run_analysis('train', args.num_train_seq, train_indices, train_data_root)
        overall_results['train_metrics'] = results
        overall_results['train_seq_indices'] = train_indices
    if do_val:
        results = run_analysis('val', args.num_val_seq, val_indices, val_data_root)
        overall_results['val_metrics'] = results
        overall_results['val_seq_indices'] = val_indices
        
    if args.name is not None:
        name = f'_{args.name}'
    else:
        name = '_'
    metrics_path = os.path.join(log_dir, f'metrics{name}_{train_step}.pkl')
    print(f'Saving metrics at {metrics_path}')
    with open(metrics_path, 'wb') as f:
        pickle.dump(overall_results, f)
