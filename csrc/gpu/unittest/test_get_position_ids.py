# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
from paddlenlp_ops import get_position_ids_and_mask_encoder_batch


class TestGetPositionIdsAndMaskEncoderBatch(unittest.TestCase):
    def setUp(self):
        pass

    def run_test_case(self, seq_lens_encoder, seq_lens_decoder, seq_lens_this_time):
        expected_pos, expected_mask = self.compute_positions_cpu(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time
        )

        seq_enc_t = paddle.to_tensor(seq_lens_encoder, dtype="int32")
        seq_dec_t = paddle.to_tensor(seq_lens_decoder, dtype="int32")
        seq_this_t = paddle.to_tensor(seq_lens_this_time, dtype="int32")

        total_length = sum(seq_lens_this_time)

        position_ids = paddle.zeros([total_length], dtype="int32")
        mask_encoder = paddle.zeros([total_length], dtype="int32")

        get_position_ids_and_mask_encoder_batch(seq_enc_t, seq_dec_t, seq_this_t, position_ids, mask_encoder)

        np_pos = position_ids.numpy()
        np_mask = mask_encoder.numpy()

        print(f"seq_enc_t: {seq_enc_t}")
        print(f"seq_dec_t: {seq_dec_t}")
        print(f"seq_this_t: {seq_this_t}")
        print(f"expected_pos: {expected_pos}")
        print(f"expected_mask: {expected_mask}")
        np.testing.assert_array_equal(np_pos, expected_pos)
        np.testing.assert_array_equal(np_mask, expected_mask)

    def compute_positions_cpu(self, seq_enc, seq_dec, seq_tt):
        total_length = sum(seq_tt)
        bsz = len(seq_enc)
        expected_pos = np.zeros((total_length), dtype=np.int32)
        expected_mask = np.zeros((total_length), dtype=np.int32)
        offset = 0
        for i in range(bsz):
            if seq_enc[i] > 0:
                for pos in range(seq_enc[i]):
                    expected_pos[offset] = pos
                    expected_mask[offset] = 1
                    offset += 1
            elif seq_dec[i] > 0:
                for pos in range(seq_tt[i]):
                    expected_pos[offset] = seq_dec[i] + pos
                    expected_mask[offset] = 0
                    offset += 1
        return expected_pos, expected_mask

    def test_1(self):
        # 1 enc 1 dec
        seq_enc = [3, 0]
        seq_dec = [0, 4]
        seq_tt = [3, 5]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_2(self):
        # all enc
        seq_enc = [2, 3]
        seq_dec = [0, 0]
        seq_tt = [2, 3]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_3(self):
        # all enc
        seq_enc = [2, 3, 1]
        seq_dec = [0, 0, 0]
        seq_tt = [2, 3, 1]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_4(self):
        # insert. enc + dec
        seq_enc = [0, 0, 5]
        seq_dec = [6, 5, 0]
        seq_tt = [1, 1, 5]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_5(self):
        # all decoder with some stop
        seq_enc = [0, 0, 0, 0, 0]
        seq_dec = [6, 5, 0, 3, 10]
        seq_tt = [1, 1, 0, 1, 1]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_6(self):
        # all decoder with some stop
        seq_enc = [0, 0, 0, 0, 0]
        seq_dec = [6, 5, 0, 3, 0]
        seq_tt = [1, 1, 0, 1]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_7(self):
        # enc + dec with some stop
        seq_enc = [0, 0, 3, 0, 0]
        seq_dec = [6, 5, 0, 3, 0]
        seq_tt = [1, 1, 3, 2]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_8(self):
        # enc + dec with some stop
        seq_enc = [0, 0, 2, 0, 0]
        seq_dec = [6, 5, 0, 0, 0]
        seq_tt = [2, 2, 2, 0, 0]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_9(self):
        # enc + dec with some stop
        seq_enc = [0, 2, 2, 0, 0]
        seq_dec = [0, 0, 0, 0, 0]
        seq_tt = [0, 2, 2, 0, 0]
        self.run_test_case(seq_enc, seq_dec, seq_tt)

    def test_10(self):
        # enc + dec with some stop
        seq_enc = [5, 0, 0, 0, 0]
        seq_dec = [0, 0, 100, 0, 0]
        seq_tt = [5, 0, 2, 0, 0]
        self.run_test_case(seq_enc, seq_dec, seq_tt)


if __name__ == "__main__":
    unittest.main()
