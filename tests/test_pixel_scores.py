import numpy as np

from model import SOCS


def test_pixel_scores_include_first_ground_truth_object():
    weights = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    instance_oh = np.array([[1], [1], [0], [0]])

    scores = SOCS.show_pixel_scores(None, weights, instance_oh, (1, 2, 2))

    assert np.any(scores[0, 0] > 0)
    assert np.any(scores[0, 1] > 0)
    np.testing.assert_array_equal(scores[1], np.zeros((2, 3)))


def test_positive_object_ids_do_not_wrap_to_background():
    instance_oh = np.eye(256, dtype=np.uint8)
    weights = np.eye(256)

    scores = SOCS.show_pixel_scores(None, weights, instance_oh, (1, 16, 16))

    assert np.all(np.any(scores > 0, axis=-1))
