import numpy as np
import torch

from segmentation_metrics import get_centroid_matches


def test_matches_return_original_slot_ids_after_filtering_empty_slots():
    predicted = torch.zeros(4, 1, 1, 2, 2)
    predicted[1, 0, 0, 0] = 1
    predicted[3, 0, 0, 1] = 1
    target = torch.zeros(3, 1, 1, 2, 2)
    target[0, 0, 0, 1] = 1
    target[2, 0, 0, 0] = 1

    distances, predicted_ids, target_ids = get_centroid_matches(predicted, target)

    np.testing.assert_allclose(distances, [0, 0])
    np.testing.assert_array_equal(predicted_ids, [3, 1])
    np.testing.assert_array_equal(target_ids, [0, 2])


def test_matches_return_only_assigned_ground_truth_ids():
    predicted = torch.ones(1, 1, 1, 2, 2)
    target = torch.zeros(2, 1, 1, 2, 2)
    target[0, 0, 0, 0, 0] = 1
    target[1] = 1

    distances, predicted_ids, target_ids = get_centroid_matches(predicted, target)

    assert len(distances) == len(predicted_ids) == len(target_ids) == 1
    np.testing.assert_array_equal(target_ids, [1])
