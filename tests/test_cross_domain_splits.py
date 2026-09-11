import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from data.build_group_splits import build


class CrossDomainSplitTest(unittest.TestCase):
    def test_device_group_is_held_out_before_windowing(self):
        with tempfile.TemporaryDirectory() as root:
            input_root = os.path.join(root, "continuous")
            output_root = os.path.join(root, "cross-device")
            os.makedirs(input_root)

            block = 80
            device = np.repeat(["d0", "d1"], block)
            subject = np.repeat(["s0", "s1"], block)
            recording = np.repeat(["r0", "r1"], block)
            y = np.concatenate(
                [np.repeat([0, 1], block // 2), np.repeat([0, 1], block // 2)]
            ).astype(np.int64)
            x = np.arange(len(y) * 2, dtype=np.float32).reshape(len(y), 2)
            for name, values in {
                "x": x,
                "y": y,
                "device": device,
                "subject": subject,
                "recording": recording,
            }.items():
                np.save(os.path.join(input_root, f"{name}.npy"), values)

            args = SimpleNamespace(
                input_root=input_root,
                output_root=output_root,
                axis="device",
                shots=[1],
                support_seeds=[0],
                validation_fraction=0.2,
                partition_seed=3,
                win_len=20,
                stride=10,
            )
            build(args)

            folds = sorted(os.listdir(output_root))
            self.assertIn("fold-0-d0", folds)
            fold = os.path.join(output_root, "fold-0-d0")
            self.assertTrue(os.path.isfile(os.path.join(fold, "x_test.npy")))
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        fold, "1-shot", "support-seed-0", "train_indices.npy"
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
