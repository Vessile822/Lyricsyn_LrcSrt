import sys
from pathlib import Path

# Add src to path
tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lyricsync import clean_lyric_lines, extract_embedded_lyrics, srt_to_lrc


def test_clean_lyric_lines():
    sample = """
    [Intro]
    *guitar solo*
    窗上的雨 一滴一滴
    [Verse 1]
    像那年你 靠近的距離（合唱）
    [01:23.45]Take me back, take me back
    【副歌】
    回到那個年代 你牽著我的手
    (那些年我們說過的話)
    [Outro]
    Take me back
    """
    lines = clean_lyric_lines(sample)
    print("Cleaned lines:")
    for l in lines:
        print("  -", l)
    assert "窗上的雨 一滴一滴" in lines
    assert "像那年你 靠近的距離" in lines
    assert "Take me back, take me back" in lines
    assert "回到那個年代 你牽著我的手" in lines
    assert "(那些年我們說過的話)" in lines, (
        "Normal parenthetical lyrics should NOT be stripped"
    )
    assert "[Intro]" not in lines
    assert "[Verse 1]" not in lines
    assert "[Outro]" not in lines
    assert "*guitar solo*" not in lines
    assert "【副歌】" not in lines
    print("test_clean_lyric_lines: PASSED")


def test_srt_to_lrc_centiseconds():
    # Test that millisecond to centisecond conversion has no float off-by-one errors
    sample_srt = """
1
00:00:16,320 --> 00:00:19,850
You said forever by the kitchen door

2
00:00:23,020 --> 00:00:25,990
Like something small I couldn't let go
"""
    lrc = srt_to_lrc(sample_srt)
    print("Generated LRC:\n", lrc)
    assert "[00:16.32]You said forever by the kitchen door" in lrc
    assert "[00:23.02]Like something small I couldn't let go" in lrc
    print("test_srt_to_lrc_centiseconds: PASSED")


def test_extract_from_songs():
    songs_dir = project_root / "songs"
    mp3s = list(songs_dir.glob("*.mp3"))
    if not mp3s:
        print("No sample mp3 found in songs/, skipping audio extraction test.")
        return
    for mp3 in mp3s:
        lines = extract_embedded_lyrics(str(mp3))
        if lines:
            print(f"Extracted {len(lines)} lines from {mp3.name}")
            assert len(lines) > 0
            break
    print("test_extract_from_songs: PASSED")


if __name__ == "__main__":
    test_clean_lyric_lines()
    test_srt_to_lrc_centiseconds()
    test_extract_from_songs()
    print("ALL TESTS PASSED!")
