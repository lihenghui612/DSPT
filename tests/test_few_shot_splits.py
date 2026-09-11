import json
import os
import tempfile
import unittest

import numpy as np

from data.build_splits import build


class FewShotSplitTest(unittest.TestCase):
    def test_support_and_validation_are_complementary(self):
        with tempfile.TemporaryDirectory() as base:
            root = os.path.join(base, "HHAR")
            os.makedirs(root)

            y_train = np.repeat(np.arange(3, dtype=np.int64), 6)
            x_train = np.arange(len(y_train) * 4, dtype=np.float32).reshape(
                len(y_train), 2, 2
            )
            np.save(os.path.join(root, "x_train.npy"), x_train)
            np.save(os.path.join(root, "y_train.npy"), y_train)
            np.save(os.path.join(root, "x_test.npy"), x_train[:3])
            np.save(os.path.join(root, "y_test.npy"), y_train[:3])

            build(
                base,
                "HHAR",
                shots=[2],
                support_seeds=[7],
                allow_unverified_root=True,
            )

            run = os.path.join(root, "2-shot", "support-seed-7")
            support = np.load(os.path.join(run, "train_indices.npy"))
            validation = np.load(os.path.join(run, "valid_indices.npy"))

            self.assertEqual(len(support), 6)
            self.assertEqual(len(validation), len(y_train) - len(support))
            self.assertEqual(len(np.intersect1d(support, validation)), 0)
            np.testing.assert_array_equal(
                np.sort(np.concatenate((support, validation))), np.arange(len(y_train))
            )
            for label in np.unique(y_train):
                self.assertEqual(int(np.sum(y_train[support] == label)), 2)

            with open(os.path.join(run, "split_manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertIn("all source indices not selected", manifest["validation"])


if __name__ == "__main__":
    unittest.main()
