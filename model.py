import math
import matplotlib.colors as pltcolors
import matplotlib.cm as pltcm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule
from torch.distributions.normal import Normal

from nets import fc_net, CNNEncoder, TransformerBlock, QueryDecoder
from segmentation_metrics import adjusted_rand_index, closest_centroids_metric
from util import MASK_COLORS

# Note: only square images for now
class SOCS(LightningModule):
    def __init__(self, 
                 img_dim_hw=(64,64), 
                 embed_dim=32,
                 num_transformer_heads=4, 
                 transformer_head_dim=128,
                 transformer_hidden_dim=1024, 
                 num_transformer_layers=4,
                 decoder_hidden_dim=1536, 
                 num_decoder_layers=3,
                 spatial_patch_hw=(6, 14),
                 num_object_slots=21, 
                 beta=1e-6,
                 bc_loss_weight=1e-4,
                 sigma_x=0.08, 
                 num_gaussian_heads=3,
                 viewpoint_size=12,
                 learning_rate=1e-4,
                 bc_task = True,
                 bc_task_transformer_layers = 2,
                 num_target_points=16,

                 # This and below are just here so they're saved w/ other hyperparameters
                 provide_viewpoint=True,
                 sequence_len=8, 
                 seed=1,
                 pixel_downsample_factor=None,
                 num_fourier_bands=10,
                 fourier_sampling_rate=60,
                 dataset_name=None,
                 dataset_root=None,
                 cameras=None,
                 ):

        super(SOCS, self).__init__()
        self.save_hyperparameters()
        self.transformer_dim = num_transformer_heads*transformer_head_dim

        self.encoder = CNNEncoder(img_dim_hw=img_dim_hw, output_dim=self.transformer_dim-self.hparams.viewpoint_size)

        transformer_1_layers = [
            TransformerBlock(self.transformer_dim, num_transformer_heads, dim_feedforward=transformer_hidden_dim, dropout=0)
            for _ in range(num_transformer_layers)]
        self.transformer_1 = nn.Sequential(*transformer_1_layers)

        num_patches = spatial_patch_hw[0] * spatial_patch_hw[1]
        if num_patches == num_object_slots:
            self.spatial_pool = nn.Identity()
        elif num_patches == 4 * num_object_slots:
            self.spatial_pool = nn.AvgPool2d((2,2), (2,2))
        else:
            raise ValueError(f'No pooling implementation for {num_patches} patches and {num_object_slots} objects')
        
        transformer_2_layers = [
            TransformerBlock(self.transformer_dim, num_transformer_heads, dim_feedforward=transformer_hidden_dim, dropout=0)
            for _ in range(num_transformer_layers)]
        self.transformer_2 = nn.Sequential(*transformer_2_layers)

        # Decode transformer output into an object latent
        self.latent_decoder = fc_net(
            num_layers=1, in_size=self.transformer_dim, hidden_size=1024, out_proj_size=2*self.hparams.embed_dim)

        # Use object latents to render specific pixels
        self.query_decoder = QueryDecoder(
            input_size=self.hparams.embed_dim, query_size=self.hparams.viewpoint_size,
            hidden_size=decoder_hidden_dim, output_size=(self.hparams.num_gaussian_heads*4)+1, num_hidden_layers=num_decoder_layers)

        self.kl_prior = Normal(torch.tensor(0), torch.tensor(1))

        if bc_task:
            task_output_dim = num_target_points * 2
            task_transformer_layers = [
                TransformerBlock(self.transformer_dim, num_transformer_heads,
                                 dim_feedforward=transformer_hidden_dim, dropout=0)
                for _ in range(num_transformer_layers)]
            self.task_transformer = nn.Sequential(*task_transformer_layers)
            self.task_mlp = fc_net(num_layers=1, in_size=self.transformer_dim, hidden_size=1024, out_proj_size=task_output_dim)

        # How many pixels should be decoded at a time when aggregating reconstructions for an entire image sequence
        self.inference_parallel_pixels = 10000

    # Dimension keys: B - batch size, T - number of frames, Y - image pixel height, X - image pixel width
    # U - number of spatial patches (height), V - number of spatial patches (width)
    # E - embedding dim, N - number of pixels to decode across all frames in sequence, K - number of object slots
    # S - viewpoint supervision dimension
    def forward(self, data):
        slot_tokens = self.get_slot_tokens(data)
        return self.decode_latents(data, slot_tokens)

    def get_slot_tokens(self, data):
        x = data['img_seq'] # \in B x T x Y x X x 3
        positional_embeddings = data['patch_positional_embeddings'] # \in B x T x Y x X x S
        batch_size = x.shape[0]
        num_frame_slots = x.shape[1]

        # Encode the entire sequence of images
        x = self.encoder(x) # \in B x T x U x V x E-S
        x = torch.cat((x, positional_embeddings), -1).flatten(1, 3) # \in B x T*U*V x E

        # Transformers
        x = x.moveaxis(0, 1) # \in T*U*V x B x E
        x = self.transformer_1(x)
        # unflatten back to T x U x V x B x E
        x = x.reshape((num_frame_slots, self.hparams.spatial_patch_hw[0], self.hparams.spatial_patch_hw[1], batch_size, self.transformer_dim))
        x = x.moveaxis(1, -1).moveaxis(1, -1) # \in T x B x E x U x V
        x = x.flatten(0, 1) # \in T*B x E x U x V
        x = self.spatial_pool(x).mul(2) # \in T*B x E x sqrt(K) x sqrt(K)
        x = x.reshape((num_frame_slots, batch_size) + x.shape[1:]) # \in T x B x E x sqrt(K) x sqrt(K)
        x = x.flatten(3, 4) # \in T x B x E x K
        x = x.moveaxis(-1, 1) # \in T x K x B x E
        x = x.flatten(0, 1) # \in T*K x B x E
        x = self.transformer_2(x)
        x = x.moveaxis(1, 0) # \in B x T*K x E
        x = x.reshape((batch_size, num_frame_slots, self.hparams.num_object_slots, self.transformer_dim)) # Uncollapse the patches

        # Aggregate along the frame latents and get the object latents
        slot_tokens = torch.mean(x, 1) # \in B x K x E
        return slot_tokens

    def decode_latents(self, data, slot_tokens, eval=False):
        output = {}
        decoder_queries = data['decoder_queries'] # \in B x N x S
        object_latent_pars = self.latent_decoder(slot_tokens)
        object_latent_mean = object_latent_pars[..., :self.hparams.embed_dim] # \in B x K x E

        if eval:
            object_latents = object_latent_mean
        else:
            object_latent_var = nn.functional.softplus(object_latent_pars[..., self.hparams.embed_dim:])
            # Sample object latents from gaussian distribution
            object_latent_distribution = Normal(object_latent_mean, object_latent_var)
            object_latents = object_latent_distribution.rsample()

        # Use queries and object latents to decode the selected pixels for loss calculations
        queries = decoder_queries.unsqueeze(1).tile(1, self.hparams.num_object_slots, 1, 1) # \in B x K x N x S
        x = self.query_decoder(object_latents, queries) # \in B x K x N x (M*4)+1
        per_object_preds = x[..., :3*self.hparams.num_gaussian_heads].unflatten(-1, (self.hparams.num_gaussian_heads, 3)) # \in B x K x N x M x 3
        per_mode_log_weights = nn.functional.log_softmax(x[..., 3*self.hparams.num_gaussian_heads : -1], -1) # \in B x K x N x M
        per_object_log_weights = nn.functional.log_softmax(x[..., -1], 1) # \in B x K x N

        # Independent gaussians for M values of R, M values of G, M values of B
        per_object_pixel_distributions = Normal(per_object_preds, self.hparams.sigma_x)
        ground_truth_rgb = data['ground_truth_rgb'].unsqueeze(1).unsqueeze(3) # \in B x 1 x N x 1 x 3

        per_object_pixel_log_likelihoods = per_object_pixel_distributions.log_prob(ground_truth_rgb) # \in B x K x N x M x 3
        # Sum across RGB because we assume the probabilities of each channel are independent, so log(P(R,G,B)) = log(P(R)P(G)P(B)) = log(P(R)) + log(P(G)) + log(P(B))
        per_object_pixel_log_likelihoods = per_object_pixel_log_likelihoods.sum(-1) # \in B x K x N x M
        # First sum the likelihoods and weights - this is equivalent to mulitplying them in regular space
        # Then apply logsumexp to add the weighted likelihoods in regular space and then convert back to log space
        weighted_mixture_log_likelihood = torch.logsumexp(per_object_pixel_log_likelihoods + per_mode_log_weights, -1) # \in B x K x N
        weighted_mixture_log_likelihood = torch.logsumexp(weighted_mixture_log_likelihood + per_object_log_weights, 1) # \in B x N

        # If using semantic segmentation to mask out the background for the reconstruction loss, do that here
        reconstruction_loss = -(weighted_mixture_log_likelihood).mean()
        output['reconstruction_loss'] = reconstruction_loss

        if self.hparams.bc_task and 'bc_waypoints' in data:
            task_tokens = self.task_transformer(slot_tokens.swapaxes(0, 1)) # \in K x B x E
            task_preds = self.task_mlp(task_tokens.mean(0)) # \in B x task_dim
            bc_mask = data['bc_mask'].unsqueeze(-1)
            targets = (data['bc_waypoints'] * bc_mask).flatten(1)
            preds = (task_preds.unflatten(-1, (-1, 2)) * bc_mask).flatten(1)
            bc_loss = nn.functional.smooth_l1_loss(preds, targets)
            output['bc_loss'] = bc_loss
            output['bc_waypoints'] = task_preds.unflatten(-1, (-1, 2))

        if eval:
            per_mode_weights = torch.exp(per_mode_log_weights) # \in B x K x N x M
            unimodal_per_object_preds = per_object_preds.mul(per_mode_weights.unsqueeze(-1)).sum(-2) # \in B x K x N x 3
            output['per_object_preds'] = unimodal_per_object_preds
            per_object_weights = torch.exp(per_object_log_weights) # \in B x K x N
            output['per_object_weights'] = per_object_weights
        else:
            # Sum across the object latent dimension, and across all objects
            kl_loss = torch.distributions.kl_divergence(object_latent_distribution, self.kl_prior).sum((1,2)).mean()
            output['kl_loss'] = kl_loss

        return output

    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.log('reconstruction_loss', output['reconstruction_loss'])
        self.log('distribution_loss', output['kl_loss'])
        loss = (output['reconstruction_loss']
                + output['kl_loss'].mul(self.hparams.beta))
        
        if self.hparams.bc_task:
            self.log('bc_loss', output['bc_loss'])
            loss += output['bc_loss'].mul(self.hparams.bc_loss_weight)

        self.log('total_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.inference_and_metrics(batch)

    def inference_and_metrics(self, batch, batch_ind=0):
        num_pix = self.inference_parallel_pixels
        pixel_ind = 0
        (B, F, H, W, _) = batch['img_seq'].shape
        (B, total_num_pix) = batch['decoder_queries'].shape[:2]

        # Predictions for all pixels at once consumes too much memory
        # So only predict num_pix at a time and aggregate results
        per_object_preds = batch['img_seq'].new_zeros(B, self.hparams.num_object_slots, total_num_pix, 3)
        per_object_weights = batch['img_seq'].new_zeros(B, self.hparams.num_object_slots, total_num_pix)
        slot_tokens = self.get_slot_tokens(batch)
        while pixel_ind < total_num_pix:
            # Not really a minibatch. But a mini batch
            minibatch = {}
            minibatch['decoder_queries'] = batch['decoder_queries'][:, pixel_ind : pixel_ind + num_pix]
            minibatch['ground_truth_rgb'] = batch['ground_truth_rgb'][:, pixel_ind : pixel_ind + num_pix]
            if 'bc_waypoints' in batch:
                minibatch['bc_waypoints'] = batch['bc_waypoints']
                minibatch['bc_mask'] = batch['bc_mask']

            mini_output = self.decode_latents(minibatch, slot_tokens, eval=True)
            per_object_preds[:, :, pixel_ind : pixel_ind + num_pix] = mini_output['per_object_preds'].detach()
            per_object_weights[:, :, pixel_ind : pixel_ind + num_pix] = mini_output['per_object_weights'].detach()
            num_pix = min(num_pix, total_num_pix - pixel_ind)
            pixel_ind += num_pix

        preds = self.mixture_preds(per_object_preds, per_object_weights)
        greedy_preds = self.greedy_preds(per_object_preds, per_object_weights)[batch_ind].cpu().detach().numpy()

        pred_rgb = preds[batch_ind].cpu().detach().numpy()
        pred_weights_tensor = per_object_weights[batch_ind].cpu().detach()
        
        ground_truth_rgb = batch['ground_truth_rgb'][batch_ind].cpu().numpy()
        reconstruction_err = np.mean((pred_rgb - ground_truth_rgb)**2)

        decode_dims = tuple(batch['decode_dims'][batch_ind].cpu())
        
        results_dict = dict(
            reconstruction_err = reconstruction_err,
            preds = pred_rgb,
            greedy_preds = greedy_preds,
            per_object_weights = pred_weights_tensor.numpy())

        # Calculate segmentation metrics when ground truth instance segmentation is available
        if 'instance_oh' in batch:
            ground_truth_segmentation_tensor = batch['instance_oh'][batch_ind].cpu().unsqueeze(0)
            if len(ground_truth_segmentation_tensor.flatten()) > 0:
                pred_segmentation_tensor = pred_weights_tensor.swapaxes(0, 1).unsqueeze(0)
                seq_ari = adjusted_rand_index(ground_truth_segmentation_tensor, pred_segmentation_tensor).numpy().item()
            else:
                seq_ari = float('nan')
            results_dict['seq_ari'] = seq_ari

            instance_mask = batch['instance_mask'][batch_ind].cpu().numpy()
            if np.any(instance_mask):
                instance_reconstruction_err = np.mean((pred_rgb[instance_mask] - ground_truth_rgb[instance_mask])**2)
            else:
                instance_reconstruction_err = float('nan')
            results_dict['instance_reconstruction_err'] = instance_reconstruction_err

            # Per-frame ARI
            frame_dims = (decode_dims[0], decode_dims[1]*decode_dims[2])
            pred_weights_seq_tensor = pred_weights_tensor.reshape((self.hparams.num_object_slots,) + frame_dims)
            instance_mask_seq = instance_mask.reshape(frame_dims)
            ground_truth_segmentation_seq_tensor = ground_truth_segmentation_tensor.reshape(frame_dims + (-1,))
            ari_frame = []
            for i in range(decode_dims[0]):
                instance_mask = instance_mask_seq[i]
                if np.any(instance_mask):
                    frame_weights_tensor = pred_weights_seq_tensor[:, i].swapaxes(0, 1).unsqueeze(0)
                    ground_truth_segmentation_frame_tensor = ground_truth_segmentation_seq_tensor[i].unsqueeze(0)
                    ari_frame.append(adjusted_rand_index(ground_truth_segmentation_frame_tensor, frame_weights_tensor).numpy().item())
                else:
                    ari_frame.append(float('nan'))
            results_dict['ari'] = np.nanmean(ari_frame)

            # Centroid distance metric
            T = F // len(self.hparams.cameras)
            C = len(self.hparams.cameras)
            gt_weights = batch['instance_oh'][batch_ind].reshape((T, C, H, W, -1)).moveaxis(-1, 0).cpu()
            pred_weights = per_object_weights[batch_ind].reshape((-1, T, C, H, W)).cpu()
            avg_centroid_dist = closest_centroids_metric(pred_weights, gt_weights)
            results_dict['avg_centroid_dist'] = avg_centroid_dist

        if 'bc_waypoints' in mini_output:
            results_dict['bc_waypoints'] = mini_output['bc_waypoints']
    
        return results_dict

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def mixture_preds(self, per_object_preds, per_object_weights):
        """
        For each pixel, return the weighted average of the predictions across objects.
        """
        return per_object_preds.mul(per_object_weights.unsqueeze(3)).sum(1) # \in B x N x 3

    def greedy_preds(self, per_object_preds, per_object_weights):
        """
        For each pixel, return the prediction of the object mask with the highest weight.
        """
        (num_batch, _, num_p, _) = per_object_preds.size()
        best_obj_ids = torch.argmax(per_object_weights, 1).reshape((num_batch, 1, num_p, 1))
        preds = torch.gather(per_object_preds, 1, best_obj_ids.tile(1,1,1,3)).squeeze(1)
        return preds

    def reconstruct_image(self, preds, dims, idx=0):
        """
        Show the reconstructed RGB image.
        """
        img_arr = preds.reshape(dims +  (3,))[idx]
        img_arr = np.clip(img_arr, 0, 1) * 255
        return img_arr.astype('uint8')

    def show_object_masks(self, per_object_weights, dims, idx=0):
        """
        Show the predicted segmentation masks.
        """
        best_obj_ids = np.argmax(per_object_weights, 0).reshape(dims)[idx]
        color_inds = np.mod(best_obj_ids, len(MASK_COLORS))
        mask_img = MASK_COLORS[color_inds.flatten()].reshape(dims[1:] + (3,))
        return mask_img.astype('uint8')

    def show_object_masks_foreground(self, per_object_weights, foreground_seg, dims, idx=0):
        """
        Show the predicted segmentation masks only for pixels that belong to ground-truth objects.
        """
        mask_img = self.show_object_masks(per_object_weights, dims, idx=idx)
        frame_foreground_seg = foreground_seg.reshape(dims)[idx]
        background_inds = np.logical_not(frame_foreground_seg.flatten())
        mask_arr_flat = mask_img.reshape(-1, 3)
        mask_arr_flat[background_inds, :] = [0,0,0]
        mask_img = mask_arr_flat.reshape(mask_img.shape)
        return mask_img

    def show_ground_truth_seg(self, instance_oh, dims, idx=0):
        """
        Show the ground-truth object segmentation.
        """
        frame_instance_oh = instance_oh.reshape(dims + (-1,))[idx]
        instance_seg_flat = np.zeros(dims[1:], dtype='uint8').flatten()
        n_total_ground_truth_obj = frame_instance_oh.shape[-1]
        frame_instance_oh_flat = frame_instance_oh.reshape(-1, n_total_ground_truth_obj)
        for i in range(n_total_ground_truth_obj):
            mask = np.where(frame_instance_oh_flat[:, i] == 1)
            instance_seg_flat[mask] = i + 1
        color_inds = np.mod(instance_seg_flat, len(MASK_COLORS))
        colors = np.concatenate(([[0,0,0]], MASK_COLORS))
        mask_img = colors[color_inds.flatten()].reshape(dims[1:] + (3,))
        return mask_img

    def show_pixel_scores(self, per_object_weights, instance_oh, dims, idx=0):
        """
        Assign a segmentation quality score to each pixel belonging to a ground-truth object, and
        plot.
        """
        frame_pred_seg = np.argmax(per_object_weights, 0).reshape(dims)[idx]
        frame_instance_oh = instance_oh.reshape(dims + (-1,)).astype('uint8')[idx]
        frame_instance_seg_flat = np.zeros(dims[1:], dtype='uint8').flatten()
        n_total_ground_truth_obj = frame_instance_oh.shape[-1]
        frame_instance_oh_flat = frame_instance_oh.reshape(-1, n_total_ground_truth_obj)
        for i in range(n_total_ground_truth_obj):
            mask = np.where(frame_instance_oh_flat[:, i] == 1)
            if len(mask) > 0:
                frame_instance_seg_flat[mask] = i

        foreground_inds = np.where(frame_instance_seg_flat != 0)
        ground_truth_seg_foreground = frame_instance_seg_flat[foreground_inds]
        pred_seg_foreground = frame_pred_seg.flatten()[foreground_inds]
        score_img = np.zeros((dims[1]*dims[2], 3))
        n_foreground_pixels = len(foreground_inds[0])
        if n_foreground_pixels > 0:
        
            pixel_scores = np.zeros(n_foreground_pixels)
            for pixel in range(n_foreground_pixels):
                pred_class = pred_seg_foreground[pixel]
                pred_pairs = pred_seg_foreground == pred_class
                gt_class = ground_truth_seg_foreground[pixel]
                gt_pairs = ground_truth_seg_foreground == gt_class
                true_pos = np.sum(pred_pairs & gt_pairs)
                false_pos = np.sum(pred_pairs & ~gt_pairs)
                true_neg = np.sum(~pred_pairs & ~gt_pairs)
                false_neg = np.sum(~pred_pairs & gt_pairs)
                pixel_rand = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
                pixel_scores[pixel] = pixel_rand

            norm = pltcolors.Normalize(vmin=0., vmax=1.)
            cmap = pltcm.ScalarMappable(norm=norm, cmap='cool')
            score_rgbs = [cmap.to_rgba(score)[:-1] for score in pixel_scores]
            score_img[foreground_inds] = score_rgbs

        return score_img.reshape(dims[1:] + (3,))