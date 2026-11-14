from unittest.mock import patch

import numpy as np
import pytest

from main import LocalDataset, SOCSDataset


def write_sequence(path):
    rgb = np.zeros((3, 3, 5, 7, 3), dtype=np.float32)
    labels = np.zeros(rgb.shape[:-1], dtype=np.int64)
    for t in range(3):
        for camera in range(3):
            rgb[t, camera] = (10 * t + camera) / 100
            labels[t, camera, 0, 0] = 10 * t + camera + 1
    np.savez(path / '0.npz', rgb=rgb,
             viewpoint_transform=np.zeros((3, 3, 12)),
             time=np.repeat(np.arange(3)[:, None], 3, axis=1),
             instance_seg=labels)
    return labels


def test_timestamps_follow_selected_camera_order_and_sequence_length(tmp_path):
    write_sequence(tmp_path)
    dataset = LocalDataset(2, (2, 2), str(tmp_path), camera_choice=[2, 0],
                           img_dim_hw=(5, 7), decode_pixel_downsample_factor=1)

    item = dataset._loaditem(0)

    np.testing.assert_array_equal(item['time_seq'], [0, 0, 1, 1])
    assert len(item['time_seq']) == len(item['img_seq'])


def test_instance_labels_follow_selected_frames_and_camera_order(tmp_path):
    labels = write_sequence(tmp_path)
    dataset = LocalDataset(2, (2, 2), str(tmp_path), camera_choice=[2, 0],
                           img_dim_hw=(5, 7), decode_pixel_downsample_factor=1,
                           add_instance_seg=True)

    item = dataset[0]

    selected = labels[:2, [2, 0]].reshape(-1)
    np.testing.assert_array_equal(item['instance_mask'], selected > 0)
    expected = np.stack([selected == value for value in np.unique(selected[selected > 0])], -1)
    np.testing.assert_array_equal(item['instance_oh'], expected)


@pytest.mark.parametrize('offset', [0, 2])
def test_decode_dimensions_describe_actual_sampled_grid(offset):
    dataset = SOCSDataset(1, (2, 2), '.', camera_choice=[0], img_dim_hw=(5, 7),
                          decode_pixel_downsample_factor=3)
    source = dict(img_seq=np.zeros((1, 5, 7, 3)),
                  viewpoint_seq=np.zeros((1, 12)), time_seq=np.zeros(1))

    with patch('numpy.random.randint', return_value=offset):
        item, _ = dataset._set_pixels_to_decode(source)

    expected = [1, len(range(offset, 5, 3)), len(range(offset, 7, 3))]
    np.testing.assert_array_equal(item['decode_dims'], expected)
    assert np.prod(item['decode_dims']) == len(item['ground_truth_rgb'])
