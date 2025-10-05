import unittest
from typing import Optional

import torch

from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL


def _make_sampling_info(
    xtc_threshold: float = 0.1,
    xtc_probability: float = 1.0,
    newline_id: int = -1,
    eos_ids=None,
    need_xtc_sampling: Optional[bool] = None,
):
    batch_size = 1
    if need_xtc_sampling is None:
        need_xtc_sampling = xtc_probability > 0

    return SamplingBatchInfo(
        temperatures=torch.ones((batch_size, 1), dtype=torch.float32),
        top_ps=torch.ones(batch_size, dtype=torch.float32),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size, dtype=torch.float32),
        is_all_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=4,
        device="cpu",
        xtc_thresholds=torch.tensor([xtc_threshold], dtype=torch.float32),
        xtc_probabilities=torch.tensor([xtc_probability], dtype=torch.float32),
        xtc_newline_token_ids=torch.tensor([newline_id], dtype=torch.long),
        xtc_eos_token_ids=[list(eos_ids) if eos_ids else []],
        need_xtc_sampling=need_xtc_sampling,
    )


class TestXTCSampler(unittest.TestCase):
    def test_masks_high_probability_tokens(self):
        sampling_info = _make_sampling_info(xtc_threshold=0.1, xtc_probability=1.0)
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5]], dtype=torch.float32)
        sampling_info.apply_xtc(logits)
        self.assertTrue(torch.isinf(logits[0, 0]))
        self.assertTrue(torch.isinf(logits[0, 1]))
        self.assertTrue(torch.isinf(logits[0, 2]))
        self.assertFalse(torch.isinf(logits[0, 3]))

    def test_skip_when_protected_token_would_be_removed(self):
        sampling_info = _make_sampling_info(newline_id=0)
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5]], dtype=torch.float32)
        original = logits.clone()
        sampling_info.apply_xtc(logits)
        self.assertTrue(torch.equal(logits, original))

    def test_threshold_requires_multiple_candidates(self):
        sampling_info = _make_sampling_info(xtc_threshold=0.6, xtc_probability=1.0)
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5]], dtype=torch.float32)
        original = logits.clone()
        sampling_info.apply_xtc(logits)
        self.assertTrue(torch.equal(logits, original))

    def test_probability_zero_disables_xtc(self):
        sampling_info = _make_sampling_info(xtc_probability=0.0)
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5]], dtype=torch.float32)
        original = logits.clone()
        sampling_info.apply_xtc(logits)
        self.assertTrue(torch.equal(logits, original))

    def test_need_flag_disables_even_with_probability(self):
        sampling_info = _make_sampling_info(
            xtc_probability=1.0, need_xtc_sampling=False
        )
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5]], dtype=torch.float32)
        original = logits.clone()
        sampling_info.apply_xtc(logits)
        self.assertTrue(torch.equal(logits, original))

    def test_repeat_interleave_respects_multiple_tokens(self):
        sampling_info = _make_sampling_info(xtc_probability=1.0)
        logits = torch.tensor(
            [
                [2.0, 1.5, 1.0, 0.5],
                [2.0, 1.5, 1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        sampling_info.apply_xtc(logits, num_tokens_in_batch=2)
        self.assertTrue(torch.isinf(logits[0, 0]) and torch.isinf(logits[1, 0]))
        self.assertFalse(torch.isinf(logits[0, 3]) or torch.isinf(logits[1, 3]))


if __name__ == "__main__":
    unittest.main()
