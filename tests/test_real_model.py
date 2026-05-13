"""Integration tests for real model hookup.

These tests require torch + transformers + GPU. They are skipped
automatically if the dependencies or hardware are unavailable.
"""

from __future__ import annotations

import pytest
import sys

# Skip entire module if torch/transformers not available
pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture
def has_cuda():
    """Skip tests if no CUDA GPU available."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    return True


class TestTraceRecorder:
    """Tests for moe_policylang.integrations.trace_recorder."""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_record_trace_basic(self, has_cuda):
        """Record traces from a small MoE model and verify format."""
        # TODO: Use a tiny model (e.g., Qwen1.5-MoE-A2.7B) for CI
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_trace_save_load_roundtrip(self):
        """Save and reload a trace, verify data integrity."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_trace_replay_matches_live(self, has_cuda):
        """Replayed trace produces same expert selections as live inference."""
        pass


class TestHuggingFaceIntegration:
    """Tests for real model integration via install_policy_hook."""

    @pytest.mark.skip(reason="Not yet implemented")
    def test_hook_install_mixtral(self, has_cuda):
        """Install hook on real Mixtral, verify inference still works."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_hook_captures_expert_ids(self, has_cuda):
        """Hook captures correct expert IDs matching router logits."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_output_numerically_identical(self, has_cuda):
        """Model output with hooks == without hooks (hooks are observation-only)."""
        pass

    @pytest.mark.skip(reason="Not yet implemented")
    def test_dsl_policy_drives_placement(self, has_cuda):
        """DSL policy's GPU/CPU decisions are respected by the integration."""
        pass
