import json
import os
import tempfile
import unittest

import numpy as np

from data.build_temporal_partitions import build


class TemporalPartitionTest(unittest.TestCase):
    def test_no_raw_index_crosses_a_split_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "continuous")
            output = os.path.join(root, "prepared")
            os.makedirs(source)

            length = 200
            x = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
            y = np.concatenate((np.zeros(100, dtype=np.int64),
                                np.ones(100, dtype=np.int64)))
            recording = np.concatenate((np.repeat("r0", 100), np.repeat("r1", 100)))
            np.save(os.path.join(source, "x.npy"), x)
            np.save(os.path.join(source, "y.npy"), y)
            np.save(os.path.join(source, "recording.npy"), recording)

            build(source, output, win_len=10, stride=5, test_fraction=0.2)

            occupied = {}
            for split in ("train", "test"):
                rec = np.load(os.path.join(output, f"recording_{split}.npy"))
                starts = np.load(os.path.join(output, f"start_{split}.npy"))
                ends = np.load(os.path.join(output, f"end_{split}.npy"))
                occupied[split] = {
                    (str(r), index)
                    for r, start, end in zip(rec, starts, ends)
                    for index in range(int(start), int(end))
                }

            self.assertTrue(occupied["train"].isdisjoint(occupied["test"]))

            with open(os.path.join(output, "temporal_split_manifest.json"),
                      encoding="utf-8") as stream:
                manifest = json.load(stream)
            self.assertTrue(manifest["split_before_windowing"])
            self.assertTrue(manifest["raw_partitions_are_disjoint"])


if __name__ == "__main__":
    unittest.main()
