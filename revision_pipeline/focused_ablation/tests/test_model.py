import unittest

import numpy as np

from revision_pipeline.focused_ablation.model import build_conv_models, EmbeddedLayout


class FocusedAblationModelTests(unittest.TestCase):
    def test_kernel_and_capacity_change_parameters(self):
        original = build_conv_models(36, 4, 3)
        kernel = build_conv_models(36, 4, 3, first_kernel=5)
        narrow = build_conv_models(36, 4, 3, filters=(16, 32, 64))
        self.assertLess(kernel[0].count_params(), original[0].count_params())
        self.assertLess(narrow[0].count_params(), original[0].count_params())
        self.assertEqual(original[0].output_shape, kernel[0].output_shape)

    def test_embedded_layout_centering_and_amplitude(self):
        layout = EmbeddedLayout(12)
        small = np.ones((2, 12, 12, 1))
        layout.inner.transform = lambda values, feature_ids: small
        output = layout.transform(np.ones((2, 1)), feature_ids=["x"])
        self.assertEqual(output.shape, (2, 36, 36, 1))
        self.assertEqual(np.count_nonzero(output[0]), 144)
        self.assertEqual(float(output.max()), 9.0)


if __name__ == "__main__": unittest.main()

