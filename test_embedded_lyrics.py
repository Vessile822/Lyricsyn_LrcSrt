import os
import sys
from pathlib import Path

# Add lyricsync to path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from lyricsync import clean_lyric_lines, fix_mojibake, extract_embedded_lyrics, find_pairs

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
    assert "[Intro]" not in lines
    assert "[Verse 1]" not in lines
    assert "[Outro]" not in lines
    assert "*guitar solo*" not in lines
    assert "【副歌】" not in lines
    print("test_clean_lyric_lines: PASSED")

def test_extract_from_real_mp3():
    sample_mp3 = Path(r"C:\Users\Vess\.openclaw\workspace\music_pipeline\inputs\audio\回到那個年代.mp3")
    if not sample_mp3.exists():
        print("Sample mp3 not found, skipping real extraction test.")
        return
    lines = extract_embedded_lyrics(str(sample_mp3))
    assert lines is not None and len(lines) > 0, "Failed to extract embedded lyrics"
    print(f"Extracted {len(lines)} lines from real mp3:")
    print("First 3 lines:", lines[:3])
    print("test_extract_from_real_mp3: PASSED")

if __name__ == "__main__":
    test_clean_lyric_lines()
    test_extract_from_real_mp3()
    print("ALL TESTS PASSED!")
