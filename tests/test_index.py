import unittest
from collections import Counter
from pathlib import Path

from tdnirs_posterior.data import (
    DatasetIndex,
    SampleRecord,
    manifest_policy_errors,
    prepare_dataset_index,
)


class DatasetIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1] / "Data"
        cls.index = DatasetIndex.discover(cls.root)

    def test_all_samples_are_kept(self) -> None:
        self.assertEqual(len(self.index.records), 6000)
        status = Counter(record.status for record in self.index.records)
        self.assertEqual(status[1], 5974)
        self.assertEqual(status[3], 26)

    def test_split_is_exact_and_reproducible(self) -> None:
        target = (4800, 600, 600)
        first = self.index.stratified_split(seed=20260907, target=target)
        second = self.index.stratified_split(seed=20260907, target=target)
        counts = Counter(record.split for record in first.records)
        self.assertEqual(counts, {"train": 4800, "validation": 600, "test": 600})
        mapping_a = {record.sample_id: record.split for record in first.records}
        mapping_b = {record.sample_id: record.split for record in second.records}
        self.assertEqual(mapping_a, mapping_b)
        self.assertEqual(set(mapping_a), set(mapping_b))

    def test_prepare_index_keeps_all_samples(self) -> None:
        data_config = {"split": [0.8, 0.1, 0.1]}
        first = prepare_dataset_index(self.root, data_config, seed=20260907)
        second = prepare_dataset_index(self.root, data_config, seed=20260907)

        self.assertEqual(len(first.records), 6000)
        self.assertEqual(
            Counter(record.layer_count for record in first.records),
            {1: 2002, 2: 2000, 3: 1998},
        )
        self.assertEqual(
            Counter(record.split for record in first.records),
            {"train": 4800, "validation": 600, "test": 600},
        )
        self.assertEqual(
            Counter(
                (record.split, record.layer_count) for record in first.records
            ),
            {
                ("train", 1): 1602,
                ("train", 2): 1600,
                ("train", 3): 1598,
                ("validation", 1): 200,
                ("validation", 2): 200,
                ("validation", 3): 200,
                ("test", 1): 200,
                ("test", 2): 200,
                ("test", 3): 200,
            },
        )

        mapping_a = {record.sample_id: record.split for record in first.records}
        mapping_b = {record.sample_id: record.split for record in second.records}
        self.assertEqual(mapping_a, mapping_b)

    def test_fixed_manifest_is_validated_semantically(self) -> None:
        data_config = {"split": [0.8, 0.1, 0.1]}
        manifest = prepare_dataset_index(self.root, data_config, seed=20260907)
        self.assertEqual(
            manifest_policy_errors(self.index, manifest, data_config), []
        )

        first = manifest.records[0]
        invalid = DatasetIndex(
            [
                SampleRecord(**{**first.__dict__, "split": "test"}),
                *manifest.records[1:],
            ],
            self.root,
        )
        self.assertTrue(
            any(
                error.startswith("manifest split counts are")
                for error in manifest_policy_errors(self.index, invalid, data_config)
            )
        )


if __name__ == "__main__":
    unittest.main()
