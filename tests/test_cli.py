"""Tests for the moe-policylang CLI."""

import pytest
from pathlib import Path

from moe_policylang.cli import main

EXAMPLES = Path(__file__).parent.parent / "examples"


class TestValidateCommand:
    def test_single_valid_file(self, capsys):
        rc = main(["validate", str(EXAMPLES / "lru_policy.moe")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "lru_baseline" in out
        assert "1 policy" in out

    def test_multiple_valid_files(self, capsys):
        files = [
            str(EXAMPLES / "lru_policy.moe"),
            str(EXAMPLES / "lfu_policy.moe"),
            str(EXAMPLES / "composed_policy.moe"),
        ]
        rc = main(["validate"] + files)
        assert rc == 0
        out = capsys.readouterr().out
        assert "3/3 files passed" in out

    def test_missing_file_fails(self, capsys):
        rc = main(["validate", "nonexistent.moe"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err

    def test_mixed_valid_and_invalid(self, capsys):
        files = [
            str(EXAMPLES / "lru_policy.moe"),
            "nonexistent.moe",
        ]
        rc = main(["validate"] + files)
        assert rc == 1
        captured = capsys.readouterr()
        assert "lru_baseline" in captured.out
        assert "1 failed" in captured.err

    def test_all_example_files(self, capsys):
        files = sorted(str(f) for f in EXAMPLES.rglob("*.moe"))
        rc = main(["validate"] + files)
        assert rc == 0
        out = capsys.readouterr().out
        assert "files passed" in out


class TestParseCommand:
    def test_parse_lru(self, capsys):
        rc = main(["parse", str(EXAMPLES / "lru_policy.moe")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Policy: lru_baseline" in out
        assert "capacity=16" in out
        assert "eviction=lru" in out

    def test_parse_composed(self, capsys):
        rc = main(["parse", str(EXAMPLES / "composed_policy.moe")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "composed_showcase" in out
        assert "Monitor:" in out

    def test_parse_missing_file(self, capsys):
        rc = main(["parse", "nonexistent.moe"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not found" in err

    def test_parse_adaptive_policy(self, tmp_path, capsys):
        policy_file = tmp_path / "adapt.moe"
        policy_file.write_text("""
            version 0.6
            policy adaptive_test {
                cache { capacity = 8  eviction = lfu  frequency_decay = 0.9 }
                adapt {
                    when hit_rate < 0.4 for 100 accesses { eviction = lru }
                }
            }
        """)
        rc = main(["parse", str(policy_file)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 rule" in out
        assert "hit_rate" in out


class TestVersionCommand:
    def test_version(self, capsys):
        rc = main(["version"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "moe-policylang" in out
        assert "1.0.0-dev" in out


class TestNoCommand:
    def test_no_args_prints_help(self, capsys):
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "moe-policylang" in out.lower() or "usage" in out.lower()
