import numpy as np
import pytest

from nv_ingest.extraction_workflows.pdf.eclair_helper import pad_arrays_if_sizes_are_different


def test_pad_arrays_with_uniform_size():
    arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr2 = np.zeros((100, 100, 3), dtype=np.uint8)
    arrays = [arr1, arr2]
    padded_arrays = pad_arrays_if_sizes_are_different(arrays)
    assert all(arr.shape == (100, 100, 3) for arr in padded_arrays), "No padding should be needed"


def test_pad_arrays_with_different_sizes():
    arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr2 = np.zeros((80, 80, 3), dtype=np.uint8)
    arrays = [arr1, arr2]
    padded_arrays = pad_arrays_if_sizes_are_different(arrays)
    assert all(arr.shape == (100, 100, 3) for arr in padded_arrays), "All arrays should be padded to (100, 100, 3)"


def test_pad_arrays_padding_content():
    arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr2 = np.ones((80, 80, 3), dtype=np.uint8)  # Different content to test padding behavior
    arrays = [arr1, arr2]
    padded_arrays = pad_arrays_if_sizes_are_different(arrays)
    # Check that padding was correctly applied
    middle_content = padded_arrays[1][10:90, 10:90]  # The original array should be centered within the new one
    assert np.array_equal(middle_content, arr2), "Padded array content should match the original"


def test_pad_arrays_non_3d_arrays():
    arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr2 = np.zeros((100,), dtype=np.uint8)  # Not a 3-D array
    arrays = [arr1, arr2]
    with pytest.raises(ValueError):
        pad_arrays_if_sizes_are_different(arrays)


def test_pad_arrays_empty_input():
    arrays = []
    padded_arrays = pad_arrays_if_sizes_are_different(arrays)
    assert padded_arrays == [], "Padded arrays should be empty for empty input"
