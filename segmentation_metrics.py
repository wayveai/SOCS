import numpy as np
import torch

from scipy.optimize import linear_sum_assignment

def adjusted_rand_index(gt_mask_oh, pred_mask_prob):
    """
    Compute the adjusted Rand index (ARI). This ignores the special case where there is only a
    single ground-truth object and will return NaN in this case.
    """
    num_pred_instances = pred_mask_prob.shape[-1]
    gt_mask_oh = gt_mask_oh.type(torch.float32)
    pred_instance_ids = torch.argmax(pred_mask_prob, dim=-1)
    pred_mask_oh = torch.nn.functional.one_hot(pred_instance_ids, num_pred_instances).type(torch.float32)
    num_points = gt_mask_oh.sum(dim=[1, 2])
    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, gt_mask_oh)
    a = nij.sum(dim=1)
    b = nij.sum(dim=2)
    r_idx = torch.sum(nij * (nij - 1), dim=[1, 2])
    a_idx = torch.sum(a * (a - 1), dim=1)
    b_idx = torch.sum(b * (b - 1), dim=1)
    expected_r_idx = (a_idx * b_idx) / (num_points * (num_points - 1))
    max_r_idx = (a_idx + b_idx) / 2
    ari = (r_idx - expected_r_idx) / (max_r_idx - expected_r_idx)
    return ari

def centroid_distance(pred_trace, gt_trace):
    dist = 0
    pred_trace = pred_trace.reshape(-1, 2)
    gt_trace = gt_trace.reshape(-1, 2)
    num_frames_with_gt_obj = 0
    for t in range(pred_trace.shape[0]):
        # No penalty if object isn't in frame
        if torch.any(torch.isnan(gt_trace[t])):
            dist += 0
        # Maximum penalty if no predicted object in frame (although this almost never happens with soft weights)
        elif torch.any(torch.isnan(pred_trace[t])):
            num_frames_with_gt_obj += 1
            dist += np.linalg.norm([2, 2])
        else:
            num_frames_with_gt_obj += 1
            dist += np.linalg.norm(pred_trace[t] - gt_trace[t])
    return (dist, num_frames_with_gt_obj)

# Assumes last 2 dimensions are H and W
def get_centroids(weights):
    (H, W) = weights.shape[-2:]
    total_ob_weights = torch.sum(weights.flatten(-2), -1)
    xs = torch.tile(torch.arange(W) * (2 / (W-1)) - 1, (H, 1))
    ys = torch.tile((torch.arange(H) * (2 / (H-1)) - 1).unsqueeze(1), (1, W))
    # Weighted average of the x (y) coordinates of every pixel in the mask
    x_centroids = torch.sum((weights * xs).flatten(-2), -1) / total_ob_weights
    y_centroids = torch.sum((weights * ys).flatten(-2), -1) / total_ob_weights
    centroids = torch.cat((y_centroids.unsqueeze(-1), x_centroids.unsqueeze(-1)), -1)
    return centroids

def get_centroid_matches(pred_weights, gt_weights):
    (_, T, C, H, W) = gt_weights.shape

    # Only consider large enough objects
    total_area = T*C*H*W
    area_threshold = 0.005 / C # 0.5% of one frame as in SAVi++
    large_gt_obj = []
    large_gt_obj_inds = []
    for (i, gt_obj) in enumerate(range(gt_weights.shape[0])):
        total_obj_area = torch.sum(gt_weights[gt_obj].flatten(), -1)
        total_ratio = total_obj_area / total_area
        if total_ratio >= area_threshold:
            large_gt_obj.append(gt_obj)
            large_gt_obj_inds.append(i)

    # Only consider predicted slots that aren't "empty"
    argmaxed_pred_obj = np.unique(np.argmax(pred_weights, 0))

    num_gt_obj = len(large_gt_obj)
    num_pred_obj = len(argmaxed_pred_obj)
    obj_dists = np.zeros((num_gt_obj, num_pred_obj))
    # TODO: only calculate centroids for filtered objects/slots
    pred_centroids = get_centroids(pred_weights)
    gt_centroids = get_centroids(gt_weights)

    # Dist between centroids of pred and gt object pairs, summed across frames
    for i, gt_obj in enumerate(large_gt_obj):
        for j, pred_obj in enumerate(argmaxed_pred_obj):
            gt_trace = gt_centroids[gt_obj]
            pred_trace = pred_centroids[pred_obj]
            (dist, num_frames_with_gt_obj) = centroid_distance(pred_trace, gt_trace)
            max_dist = np.linalg.norm([2, 2]) * num_frames_with_gt_obj
            obj_dists[i, j] = dist / max_dist

    (row_ind, col_ind) = linear_sum_assignment(obj_dists)
    best_dists = obj_dists[row_ind, col_ind]
    gt_inds = np.array(large_gt_obj_inds)
    pred_inds = col_ind
    return (best_dists, pred_inds, gt_inds)

def closest_centroids_metric(pred_weights, gt_weights):
    best_dists = get_centroid_matches(pred_weights, gt_weights)[0]
    return np.mean(best_dists)