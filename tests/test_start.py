"""start.sh must not drop the kws extra while the spotter is turned on."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def _enabled(text: str, tmp_path: Path) -> bool:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    proc = subprocess.run(
        ["bash", "-c", f'source "{START}"; kws_enabled "{path}"'],
        check=False, capture_output=True, text=True)
    return proc.returncode == 0


def test_kws_extra_follows_the_config_flag(tmp_path):
    assert _enabled("[kws]\nenabled = true\n", tmp_path)
    assert _enabled("[turns]\nenabled = true\n\n[kws]\nenabled = true  # spotter\n", tmp_path)
    assert _enabled("[kws]\nenabled = false\nenabled = true\n", tmp_path)
    assert not _enabled("[kws]\nenabled = false\n", tmp_path)
    assert not _enabled("[kws]\nenabled = true\nenabled = false\n", tmp_path)
    assert not _enabled("[cue]\nenabled = true\n[turns]\nenabled = true\n", tmp_path)
    assert not _enabled("[kws]\n# enabled = true\nenabled = false\n", tmp_path)
    missing = subprocess.run(
        ["bash", "-c", f'source "{START}"; kws_enabled "{tmp_path / "missing.toml"}"'],
        check=False, capture_output=True, text=True)
    assert missing.returncode != 0


def test_sync_args_include_kws_only_when_enabled(tmp_path):
    on = tmp_path / "on.toml"
    off = tmp_path / "off.toml"
    on.write_text("[kws]\nenabled = true\n", encoding="utf-8")
    off.write_text("[audio]\ndevice = \"\"\n", encoding="utf-8")

    def args(path: Path) -> str:
        proc = subprocess.run(
            ["bash", "-c", f'source "{START}"; kws_sync_args "{path}"'],
            check=False, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    assert args(on) == "--extra dev --extra kws"
    assert args(off) == "--extra dev"
    text = START.read_text(encoding="utf-8")
    assert 'uv sync "${sync_args[@]}"' in text
    assert "uv sync --extra dev\n" not in text
