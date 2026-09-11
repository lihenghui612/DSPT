import json
import os
import tempfile
import unittest

import numpy as np

from data.build_subject_partitions import build


class SubjectPartitionTest(unittest.TestCase):
    def test_subjects_are_split_before_windowing(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "continuous")
            output = os.path.join(root, "prepared")
            os.makedirs(source)

            block = 100
            subjects = np.repeat(["s0", "s1", "s2"], block)
            recordings = np.repeat(["r0", "r1", "r2"], block)
            labels = np.repeat([0, 1, 0], block).astype(np.int64)
            readings = np.arange(len(labels) * 2, dtype=np.float32).reshape(len(labels), 2)
            np.save(os.path.join(source, "x.npy"), readings)
            np.save(os.path.join(source, "y.npy"), labels)
            np.save(os.path.join(source, "subject.npy"), subjects)
            np.save(os.path.join(source, "recording.npy"), recordings)

            build(
                source,
                output,
                win_len=20,
                stride=10,
                test_subjects=["s2"],
            )

            source_subjects = set(np.load(os.path.join(output, "subject_train.npy")))
            test_subjects = set(np.load(os.path.join(output, "subject_test.npy")))
            self.assertEqual(source_subjects, {"s0", "s1"})
            self.assertEqual(test_subjects, {"s2"})
            self.assertTrue(source_subjects.isdisjoint(test_subjects))

            with open(
                os.path.join(output, "subject_split_manifest.json"), encoding="utf-8"
            ) as stream:
                manifest = json.load(stream)
            self.assertTrue(manifest["split_before_windowing"])
            self.assertTrue(manifest["subject_partitions_are_disjoint"])
            self.assertEqual(manifest["test_subjects"], ["s2"])


if __name__ == "__main__":
    unittest.main()
