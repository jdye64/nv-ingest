# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the module under test
import nv_ingest_api.internal.primitives.nim.model_interface.yolox as model_interface_module
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
    YoloxModelInterfaceBase,
    YoloxGraphicElementsModelInterface,
)

MODULE_UNDER_TEST = f"{model_interface_module.__name__}"

# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors


class TestYoloxGraphicElementsModelInterface(unittest.TestCase):

    def setUp(self):
        # Mock constants
        self.constants_patcher = patch.multiple(
            MODULE_UNDER_TEST,
            YOLOX_GRAPHIC_NIM_MAX_IMAGE_SIZE=1000000,
            YOLOX_GRAPHIC_NUM_CLASSES=5,
            YOLOX_GRAPHIC_CONF_THRESHOLD=0.5,
            YOLOX_GRAPHIC_IOU_THRESHOLD=0.45,
            YOLOX_GRAPHIC_MIN_SCORE=0.3,
            YOLOX_GRAPHIC_FINAL_SCORE=0.3,
            YOLOX_GRAPHIC_CLASS_LABELS=["logo", "image", "diagram", "chart", "table"],
        )
        self.mocked_constants = self.constants_patcher.start()

        # Mock the logger
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        # Mock the get_bbox_dict_yolox_graphic function
        self.get_bbox_dict_patcher = patch(f"{MODULE_UNDER_TEST}.get_bbox_dict_yolox_graphic")
        self.mock_get_bbox_dict = self.get_bbox_dict_patcher.start()

        # Create sample output from get_bbox_dict_yolox_graphic
        self.sample_bbox_dict = {
            "logo": np.array([[10, 20, 30, 40, 0.9]]),
            "image": np.array([[50, 60, 70, 80, 0.85]]),
            "chart": np.array([]),
        }
        self.mock_get_bbox_dict.return_value = self.sample_bbox_dict

        # Create an instance of YoloxGraphicElementsModelInterface
        self.model_interface = YoloxGraphicElementsModelInterface()

        # Sample test data
        self.sample_annotations = [{"logo": [[0.1, 0.2, 0.3, 0.4, 0.9]]}, {"image": [[0.5, 0.6, 0.7, 0.8, 0.85]]}]
        self.sample_shapes = [(100, 200, 3), (150, 250, 3)]

    def tearDown(self):
        # Stop all patchers
        self.constants_patcher.stop()
        self.logger_patcher.stop()
        self.get_bbox_dict_patcher.stop()

    def test_initialization(self):
        """Test initialization of YoloxGraphicElementsModelInterface."""
        # Check parent class initialization
        self.assertEqual(self.model_interface.nim_max_image_size, 1000000)
        self.assertEqual(self.model_interface.num_classes, 5)
        self.assertEqual(self.model_interface.conf_threshold, 0.5)
        self.assertEqual(self.model_interface.iou_threshold, 0.45)
        self.assertEqual(self.model_interface.min_score, 0.3)
        self.assertEqual(self.model_interface.final_score, 0.3)
        self.assertEqual(self.model_interface.class_labels, ["logo", "image", "diagram", "chart", "table"])

    def test_inheritance(self):
        """Test that YoloxGraphicElementsModelInterface inherits from YoloxModelInterfaceBase."""
        self.assertIsInstance(self.model_interface, YoloxModelInterfaceBase)

    def test_name(self):
        """Test the name method."""
        self.assertEqual(self.model_interface.name(), "yolox-graphic-elements")

    def test_postprocess_annotations(self):
        """Test postprocess_annotations method."""
        # Save original transform_normalized_coordinates_to_original method
        original_transform = self.model_interface.transform_normalized_coordinates_to_original

        # Create mock for transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=self.sample_annotations
        )

        try:
            # Call postprocess_annotations
            result = self.model_interface.postprocess_annotations(
                self.sample_annotations, original_image_shapes=self.sample_shapes
            )

            # Check that transform_normalized_coordinates_to_original was called correctly
            self.model_interface.transform_normalized_coordinates_to_original.assert_called_once_with(
                self.sample_annotations, self.sample_shapes
            )

            # Check that get_bbox_dict_yolox_graphic was called for each annotation
            self.assertEqual(self.mock_get_bbox_dict.call_count, 2)

            # Check first call
            self.mock_get_bbox_dict.assert_any_call(
                self.sample_annotations[0],
                self.sample_shapes[0],
                self.model_interface.class_labels,
                self.model_interface.min_score,
            )

            # Check second call
            self.mock_get_bbox_dict.assert_any_call(
                self.sample_annotations[1],
                self.sample_shapes[1],
                self.model_interface.class_labels,
                self.model_interface.min_score,
            )

            # Check the result length
            self.assertEqual(len(result), 2)

            # Check that numpy arrays were converted to lists
            for bbox_dict in result:
                for label, bboxes in bbox_dict.items():
                    self.assertIsInstance(bboxes, list)

            # Check specific values in the result
            self.assertEqual(
                result[0]["logo"].tolist() if hasattr(result[0]["logo"], "tolist") else result[0]["logo"],
                [[10, 20, 30, 40, 0.9]],
            )
            self.assertEqual(
                result[1]["image"].tolist() if hasattr(result[1]["image"], "tolist") else result[1]["image"],
                [[50, 60, 70, 80, 0.85]],
            )
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_with_empty_shapes(self):
        """Test postprocess_annotations with empty shapes."""
        # Save original transform_normalized_coordinates_to_original method
        original_transform = self.model_interface.transform_normalized_coordinates_to_original

        # Create mock for transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=self.sample_annotations
        )

        try:
            # Call postprocess_annotations with empty shapes
            result = self.model_interface.postprocess_annotations(self.sample_annotations)

            # Check that transform_normalized_coordinates_to_original was called correctly
            self.model_interface.transform_normalized_coordinates_to_original.assert_called_once_with(
                self.sample_annotations, []
            )
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_numpy_array_conversion(self):
        """Test that postprocess_annotations correctly converts numpy arrays to lists."""
        # Mock get_bbox_dict_yolox_graphic to return dict with numpy arrays
        self.mock_get_bbox_dict.return_value = {
            "logo": np.array([[10, 20, 30, 40, 0.9]]),
            "image": np.array([[50, 60, 70, 80, 0.85]]),
        }

        # Save original transform_normalized_coordinates_to_original method
        original_transform = self.model_interface.transform_normalized_coordinates_to_original

        # Create mock for transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=[self.sample_annotations[0]]  # Just use one annotation for simplicity
        )

        try:
            # Call postprocess_annotations
            result = self.model_interface.postprocess_annotations(
                [self.sample_annotations[0]], original_image_shapes=[self.sample_shapes[0]]
            )

            # Check that the result has lists, not numpy arrays
            self.assertIsInstance(result[0]["logo"], list)
            self.assertIsInstance(result[0]["image"], list)

            # Check content of converted lists
            self.assertEqual(result[0]["logo"], [[10, 20, 30, 40, 0.9]])
            self.assertEqual(result[0]["image"], [[50, 60, 70, 80, 0.85]])
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform

    def test_postprocess_annotations_non_numpy_values(self):
        """Test that postprocess_annotations correctly handles non-numpy values."""
        # Mock get_bbox_dict_yolox_graphic to return dict with mixed types
        self.mock_get_bbox_dict.return_value = {
            "logo": [[10, 20, 30, 40, 0.9]],  # Already a list
            "image": np.array([[50, 60, 70, 80, 0.85]]),  # Numpy array
            "empty": [],  # Empty list
        }

        # Save original transform_normalized_coordinates_to_original method
        original_transform = self.model_interface.transform_normalized_coordinates_to_original

        # Create mock for transform_normalized_coordinates_to_original
        self.model_interface.transform_normalized_coordinates_to_original = MagicMock(
            return_value=[self.sample_annotations[0]]
        )

        try:
            # Call postprocess_annotations
            result = self.model_interface.postprocess_annotations(
                [self.sample_annotations[0]], original_image_shapes=[self.sample_shapes[0]]
            )

            # Check that all values are lists
            self.assertIsInstance(result[0]["logo"], list)
            self.assertIsInstance(result[0]["image"], list)
            self.assertIsInstance(result[0]["empty"], list)

            # Check that the content is preserved
            self.assertEqual(result[0]["logo"], [[10, 20, 30, 40, 0.9]])
            self.assertEqual(result[0]["image"], [[50, 60, 70, 80, 0.85]])
            self.assertEqual(result[0]["empty"], [])
        finally:
            # Restore the original method
            self.model_interface.transform_normalized_coordinates_to_original = original_transform


if __name__ == "__main__":
    unittest.main()
