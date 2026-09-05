import struct
import unittest

import numpy as np

from smoke_labeler.bag import _custom_msg_cdr_arrays


class RawDecoderTests(unittest.TestCase):
    def test_cdr_preserves_fields_and_point_timing(self):
        payload = bytearray(b"\x00\x01\x00\x00")
        payload.extend(struct.pack("<iiI", 1, 0, 6))
        payload.extend(b"livox\x00")
        payload.extend(bytes(-(len(payload) - 4) % 8))
        payload.extend(struct.pack("<QIB3sI", 1_000_000_000, 2, 0, bytes(3), 2))
        payload.extend(struct.pack("<IfffBBB", 1000, 1, 2, 3, 42, 16, 0))
        payload.extend(b"\x00")
        payload.extend(struct.pack("<IfffBBB", 2000, 4, 5, 6, 84, 32, 1))
        xyz, reflectivity, tag, line, offsets, timebase = _custom_msg_cdr_arrays(bytes(payload))
        np.testing.assert_array_equal(xyz, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(reflectivity, [42, 84])
        np.testing.assert_array_equal(tag, [16, 32])
        np.testing.assert_array_equal(line, [0, 1])
        np.testing.assert_allclose(offsets, [0.000001, 0.000002])
        self.assertEqual(timebase, 1_000_000_000)
        sampled = _custom_msg_cdr_arrays(bytes(payload), point_stride=2)
        np.testing.assert_array_equal(sampled[0], [[1, 2, 3]])

    def test_truncated_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            _custom_msg_cdr_arrays(bytes(12))
