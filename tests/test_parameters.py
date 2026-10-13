import unittest

import numpy as np

from tdnirs_posterior.parameters import ParameterSchema


class ParameterSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = ParameterSchema()

    def test_active_dimensions(self) -> None:
        self.assertEqual(int(self.schema.active_mask(1).sum()), 2)
        self.assertEqual(int(self.schema.active_mask(2).sum()), 5)
        self.assertEqual(int(self.schema.active_mask(3).sum()), 8)

    def test_normalization_round_trip(self) -> None:
        values = np.array(
            [2.0, 6.0, 0.005, 0.01, 0.02, 0.6, 1.2, 2.4], dtype=np.float32
        )
        restored = self.schema.denormalize(self.schema.normalize(values))
        np.testing.assert_allclose(restored, values, rtol=2e-6, atol=1e-7)

    def test_denormalize_does_not_silently_clip(self) -> None:
        normalized = np.zeros(8, dtype=np.float32)
        normalized[0] = 1.2
        restored = self.schema.denormalize(normalized)
        self.assertGreater(float(restored[0]), self.schema.thickness_bounds_mm[1])

    def test_positive_optical_lower_extrapolation_survives_round_trip(self) -> None:
        normalized = np.zeros(8, dtype=np.float32)
        normalized[2] = -2.0
        physical = self.schema.denormalize(normalized)
        self.assertGreater(float(physical[2]), 0.0)
        self.assertLess(float(physical[2]), self.schema.mua_bounds_mm1[0])
        restored = self.schema.normalize(physical)
        self.assertAlmostEqual(float(restored[2]), -2.0, places=5)

    def test_normalized_support_rejects_outside_and_nonfinite_values(self) -> None:
        normalized = np.zeros((3, 8), dtype=np.float32)
        normalized[1, 2] = 1.01
        normalized[2, 5] = np.nan
        valid = self.schema.normalized_valid_mask(1, normalized)
        np.testing.assert_array_equal(valid, [True, False, False])

    def test_known_mask_requires_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            self.schema.known_arrays(
                1, {"mua1_mm-1": 0.01, "musp1_mm-1": 1.0}
            )

    def test_n3_thickness_constraint(self) -> None:
        values = np.array([20.0, 20.0, 0.01, 0.01, 0.01, 1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            self.schema.validate_physical(3, values)


if __name__ == "__main__":
    unittest.main()
