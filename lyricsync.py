"""lyricsync: song + lyrics txt -> SRT
pipeline: demucs vocal separation -> Whisper (turbo) forced alignment of the given lyrics
          -> extend line ends through the sung tail -> flag suspicious lines.
usage: python lyricsync.py <song | lyrics.txt | folder> [...] [--lang ja]
"""
import os, re, time, argparse, warnings, difflib, json, traceback, shutil
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

DEV = "cuda" if torch.cuda.is_available() else "cpu"
try:
    _cfg = json.loads(Path(__file__).with_name("runtime_config.json").read_text(encoding="utf-8"))
    if _cfg.get("device") == "cpu":
        DEV = "cpu"
except (OSError, ValueError):
    pass
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False  # same input -> same SRT every run
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
WHISPER = "turbo"
_models = {}


def load_audio(path):
    try:
        x, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception:  # mp3/m4a etc via ffmpeg
        import subprocess
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "2", "-ar", "44100", "-"],
                             capture_output=True, check=True).stdout
        x, sr = np.frombuffer(raw, np.float32).reshape(-1, 2), 44100
    return x, sr


def separate_vocals(x, sr):
    """-> mono vocals at 16 kHz"""
    import gc
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    m = get_model("htdemucs").to(DEV).eval()
    if sr != m.samplerate:
        x = resample_poly(x, m.samplerate, sr, axis=0).astype(np.float32)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    wav = torch.from_numpy(x.T.copy()).to(DEV)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    import random
    random.seed(0)  # demucs picks a random time shift -> fix it so the same song always gives the same SRT
    with torch.no_grad():
        out = apply_model(m, wav[None], device=DEV, split=True, overlap=0.25, progress=False)[0]  # shifts=2 tested 2026-10-15: worse
    voc = out[m.sources.index("vocals")] * (ref.std() + 1e-8) + ref.mean()
    voc_np = voc.mean(0).cpu().numpy()
    # 立即銷毀 Demucs 模型與張量，釋放 1.5GB+ VRAM 與 RAM
    del m, wav, out, voc, ref
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return resample_poly(voc_np, 160, 441).astype(np.float32)  # 44.1k -> 16k


def fix_mojibake(text: str) -> str:
    """Recover UTF-8 text misread as Latin-1 by ffprobe (common with Korean/Japanese/Chinese tags)."""
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        if fixed != text and re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", fixed):
            return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


def clean_lyric_lines(raw_text: str) -> list:
    """Split and clean embedded or raw lyrics, stripping section markers and annotations."""
    lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        # Handle potential LRC timestamps
        line = re.sub(r"^\[\d+:\d+(?:\.\d+)?\]\s*", "", line).strip()
        # Skip pure section markers: [Chorus], [Verse 1], *guitar solo*, 【副歌】
        if not line or re.fullmatch(r"[\[\(（【\*].*[\]\)）】\*]", line):
            continue
        # Clean parenthetical annotations like （合唱）
        line = re.sub(r"[（\(][^）\)]*[）\)]", "", line).strip()
        if line:
            lines.append(line)
    return lines


def extract_embedded_lyrics(audio_path: str):
    """Extract lyrics from audio metadata tags (Suno AI, Udio, ID3, etc.)."""
    import subprocess
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format_tags", "-of", "json", audio_path],
            capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace"
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        data = json.loads(proc.stdout)
        tags = data.get("format", {}).get("tags", {})
        for k, v in tags.items():
            k_lower = k.lower()
            if "lyrics" in k_lower or "unsyncedlyrics" in k_lower:
                text = fix_mojibake(str(v))
                lines = clean_lyric_lines(text)
                if lines:
                    return lines
    except Exception:
        pass
    return None


def read_lyrics(path):
    lines = []
    for ln in open(path, encoding="utf-8-sig").read().splitlines():
        s = ln.strip()
        if not s or re.fullmatch(r"[\[\(（【\*].*[\]\)）】\*]", s):  # skip blank and [Verse]-style tags
            continue
        lines.append(s)
    return lines


def guess_lang(lines):
    t = "".join(lines)
    if re.search(r"[぀-ヿ]", t): return "ja"
    if re.search(r"[가-힯]", t): return "ko"
    if re.search(r"[一-鿿]", t): return "zh"
    return None  # let whisper detect


CJK = r"[぀-ヿ㐀-鿿가-힯]"


def units(text, lang):
    """split text into comparable units: one per CJK char (pinyin for zh), one per latin word"""
    toks = re.findall(CJK + r"|[A-Za-z0-9'’]+", text)
    norm = lambda t: re.sub(r"['’]", "", t.lower())  # don't == dont
    if lang == "zh":
        from pypinyin import lazy_pinyin
        return [lazy_pinyin(t)[0] if re.match(CJK, t) else norm(t) for t in toks]
    return [norm(t) for t in toks]


def get_whisper():
    import stable_whisper, whisper, gc
    if "whisper" not in _models:
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()
            with torch.device("cuda"):
                orig_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float16)
                try:
                    m = stable_whisper.load_model(WHISPER, device="cuda", cpu_preload=False)
                finally:
                    torch.set_default_dtype(orig_dtype)
            for mod in m.modules():
                if isinstance(mod, (torch.nn.LayerNorm, whisper.model.LayerNorm)):
                    mod.float()
            _models["whisper"] = m
        else:
            _models["whisper"] = stable_whisper.load_model(WHISPER, device="cpu", cpu_preload=True)
    return _models["whisper"]


INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")


def split_py(u):
    for ini in INITIALS:
        if u.startswith(ini):
            return ini, u[len(ini):]
    return "", u


LANG_IS_ZH = [True]


def sim(x, y):
    """2 = same unit, 0.5 = sounds alike (same pinyin initial or final), -1 = different"""
    if x == y:
        return 2.0
    if x.isascii() and y.isascii() and x.isalpha() and y.isalpha():
        (i1, f1), (i2, f2) = split_py(x), split_py(y)
        if LANG_IS_ZH[0]:
            if f1 and f1 == f2 or i1 and i1 == i2:
                return 0.5
        elif difflib.SequenceMatcher(None, x, y).ratio() >= 0.6:  # english: misheard but close spelling
            return 0.5
    return -1.0


def anchors(v16, lines, lang):
    """free transcription + DP match against lyrics -> per line (start, end, match ratio) or None"""
    torch.manual_seed(0)
    LANG_IS_ZH[0] = lang == "zh"
    r = get_whisper().transcribe(v16, language=lang, word_timestamps=True, verbose=None, fp16=(DEV == "cuda"),
                                 condition_on_previous_text=False)
    A = []  # (unit, time)
    for w in r.all_words():
        us = units(w.word, lang)
        for k, u in enumerate(us):
            A.append((u, w.start + (w.end - w.start) * (k + 0.5) / len(us)))
    L = []  # (unit, line index)
    for i, ln in enumerate(lines):
        L += [(u, i) for u in units(ln, lang)]
    n, m = len(L), len(A)
    G = -1.0
    S = np.zeros((n + 1, m + 1)); S[1:, 0] = G * np.arange(1, n + 1)  # ASR side: free leading gap
    B = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        li = L[i - 1][0]
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + sim(li, A[j - 1][0])
            u = S[i - 1, j] + G; l = S[i, j - 1] + G
            if d >= u and d >= l: S[i, j], B[i, j] = d, 0
            elif u >= l: S[i, j], B[i, j] = u, 1
            else: S[i, j], B[i, j] = l, 2
    i, j = n, int(S[n].argmax())  # free trailing gap on ASR side
    hit = {}  # lyric unit -> (time, weight)
    while i > 0 and j > 0:
        if B[i, j] == 0:
            sc = sim(L[i - 1][0], A[j - 1][0])
            if sc > 0: hit[i - 1] = (A[j - 1][1], 1.0 if sc == 2 else 0.5)
            i -= 1; j -= 1
        elif B[i, j] == 1: i -= 1
        else: j -= 1
    out = []
    for li in range(len(lines)):
        idx = [k for k in range(n) if L[k][1] == li]
        got = [(k, hit[k][0]) for k in idx if k in hit]
        ratio = sum(hit[k][1] for k in idx if k in hit) / max(1, len(idx))
        if len(got) >= 2:
            k0, k1 = idx[0], idx[-1]
            # each matched unit implies a line start; the median ignores stray matches (common chars like 的)
            imp = np.array([t - 0.25 * (k - k0) for k, t in got])
            med = float(np.median(imp))
            good = [(k, t) for (k, t), x in zip(got, imp) if abs(x - med) < 0.8]
            if not good:  # matches disagree with each other -> no anchor
                out.append(None); continue
            st = good[0][1] - 0.25 * (good[0][0] - k0) - 0.1
            en = good[-1][1] + 0.25 * (k1 - good[-1][0]) + 0.1
            ks = {k for k, _ in good}
            measured = bool(ks & set(idx[:2])) and bool(ks & set(idx[-2:]))  # start/end heard, not extrapolated
            out.append((st, en, ratio * len(good) / len(got) if measured else 0.0))
        else:
            out.append(None)
    return out


def whisper_align(v16, lines, lang, trust=0.5):
    """1) free transcription gives every line an anchor (one lost line cannot drag the rest along);
    2) runs of lines whose anchor is weak get force-aligned together, inside the gap between trusted anchors."""
    model = get_whisper()
    anc = anchors(v16, lines, lang)
    total = len(v16) / 16000
    ok = [a is not None and a[2] >= trust for a in anc]
    if not any(ok):
        raise RuntimeError("聽寫完全對不上歌詞（語言或歌詞檔錯了？）")
    spans = [(a[0], a[1], 1.0) if g else None for a, g in zip(anc, ok)]
    i = 0
    while i < len(lines):
        if ok[i]:
            i += 1; continue
        j = i
        while j < len(lines) and not ok[j]:
            j += 1
        lo = spans[i - 1][1] - 0.1 if i else None
        hi = spans[j][0] + 0.1 if j < len(lines) else None
        # song edges: search at most 8 s per line beyond the outermost trusted anchor (skips long intros/outros)
        if lo is None: lo = max(0.0, (hi if hi is not None else total) - 8.0 * (j - i))
        if hi is None: hi = min(total, lo + 8.0 * (j - i))
        block = lines[i:j]
        got = None
        if hi - lo > 0.5:
            try:
                r = model.align(v16[int(lo * 16000):int(hi * 16000)], "\n".join(block),
                                language=lang or "en", original_split=True, verbose=None)
                segs = [s for s in r.segments if s.text.strip()]
                if len(segs) == len(block):
                    got = [(lo + s.start, lo + s.end, 0.5) for s in segs]
            except Exception:
                pass
        if got is None:  # give up: spread the block evenly, flag it
            step = (hi - lo) / len(block)
            got = [(lo + step * k, lo + step * (k + 1), -1.0) for k in range(len(block))]
        spans[i:j] = got
        i = j
    # second pass: a line squeezed into a sliver (aligner glued it to the previous line) gets re-aligned alone
    for i, ln in enumerate(lines):
        st, en, p = spans[i]
        n_u = max(1, len(units(ln, lang)))
        if p >= 1.0 or (en - st) / n_u >= (0.12 if lang == "zh" else 0.18):
            continue
        lo = spans[i - 1][1] if i else max(0.0, st - 8)
        hi = spans[i + 1][0] if i + 1 < len(lines) else min(total, en + 8)
        if hi - lo < 0.5:
            continue
        try:
            r = model.align(v16[int(lo * 16000):int(hi * 16000)], ln, language=lang or "en", verbose=None)
            ws = [w for sg in r.segments for w in sg.words]
            if ws and ws[-1].end - ws[0].start > en - st:
                spans[i] = (lo + ws[0].start, lo + ws[-1].end, 0.5)
        except Exception:
            pass
    out, prev_end = [], 0.0
    for st, en, p in spans:
        st = max(st, prev_end)
        out.append((st, max(en, st + 0.2), p))
        prev_end = out[-1][1]
    return out


def extend_tails(spans, v16, drop=20, max_ext=1.0, gap=1.2):  # tuned on 5 user-checked songs
    """push each line end through its sustained vocal tail (stops before the next line)."""
    hop = 160  # 10 ms
    n = len(v16) // hop
    e = np.sqrt((v16[:n * hop].reshape(n, hop) ** 2).mean(1))
    db = 20 * np.log10(np.convolve(e, np.ones(5) / 5, "same") + 1e-9)
    voiced = db > np.percentile(db, 95) - drop  # reverb tails stay below this
    out = []
    for i, (a, b, p) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else n / 100
        # trim: if the line runs into >0.8 s of silence (instrumental break), end it at the last sung frame
        s0, k = int(a * 100), int(b * 100)
        # a start sitting in silence moves forward to the first sung frame
        f = s0
        while f < k - 20 and f < n and not voiced[f]:
            f += 1
        if f - s0 >= 15:
            s0, a = f, f / 100
        run = sung = 0
        for f in range(s0, min(k, n)):
            if voiced[f]:
                run = 0; sung += 1
            else:
                run += 1
            if run >= int(gap * 100):
                if p < 1.0 and voiced[f:min(k, n)].sum() > max(50, 2 * sung):  # only for lines not anchored by ASR
                    # silence right after a tiny bit of voice (previous line's tail) = pause before the line: move start
                    g = f
                    while g < min(k, n) and not voiced[g]:
                        g += 1
                    s0, a, run, sung = g, g / 100, 0, 0
                    continue
                k = f - run + 1  # silence after the line was sung = instrumental: cut the end
                b = k / 100
                break
        lim = min(int(nxt * 100) - 5, k + int(max_ext * 100))  # keep 50 ms gap
        while k < lim and voiced[k]:
            k += 1
        out.append((a, max(b, k / 100), p))
    return out


def fmt(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def srt_to_lrc(srt_text: str) -> str:
    """1:1 reverse-engineered implementation of ailrcgenerator.com (srt-to-lrc)."""
    blocks = re.split(r'\n\s*\n', srt_text.strip())
    lrc_lines = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) >= 3:
            time_match = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
            if time_match:
                h, m, s, ms = int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3)), int(time_match.group(4))
                total_sec = h * 3600 + m * 60 + s + ms / 1000.0
                lrc_min = int(total_sec // 60)
                rem = total_sec % 60
                lrc_sec = int(rem)
                lrc_cent = int((rem - lrc_sec) * 100)
                ts = f"[{lrc_min:02d}:{lrc_sec:02d}.{lrc_cent:02d}]"
                for text_line in lines[2:]:
                    text_clean = text_line.strip()
                    if text_clean:
                        lrc_lines.append(f"{ts}{text_clean}")
    return "\n".join(lrc_lines) + "\n"


def run(audio, lyr, outp, lang=None):
    t0 = time.time()
    lines = read_lyrics(lyr)
    lang = lang or guess_lang(lines)
    print(f"{os.path.basename(audio)}  ({len(lines)} lines, lang={lang}, {DEV})", flush=True)
    print("  loading audio...", flush=True)
    x, sr = load_audio(audio)
    print("  separating vocals (Demucs)...", flush=True)
    v16 = separate_vocals(x, sr)
    print(f"  vocals separated  {time.time()-t0:.0f}s", flush=True)
    import mms_refine
    print("  aligning lyrics (Whisper turbo)...", flush=True)
    spans = whisper_align(v16, lines, lang)
    _models.pop("whisper", None)
    import gc
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    print("  refining timestamps (MMS)...", flush=True)
    spans = extend_tails(mms_refine.refine(v16, spans, lines, DEV), v16)  # MMS sharpens starts, any language
    mms_refine._m.clear()
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    print(f"  aligned           {time.time()-t0:.0f}s", flush=True)
    srt, warn = [], []
    for i, ((a, b, p), ln) in enumerate(zip(spans, lines)):
        srt.append(f"{i+1}\n{fmt(a)} --> {fmt(b)}\n{ln}\n")
        n_units = max(1, len(re.findall(r"[぀-鿿가-힯]|[A-Za-z']+", ln)))
        per = (b - a) / n_units
        why = []
        if p < 0: why.append("對不上，時間是平均分配的")
        if per < 0.07: why.append("太短（可能被擠壓）")
        if per > 1.5: why.append("太長（可能跨到別段）")
        if i and a < spans[i - 1][1] - 0.05: why.append("和上一句重疊")
        if why:
            warn.append(f"  第{i+1}句 {fmt(a)}  {'、'.join(why)}  {ln}")

    srt_content = "\n".join(srt)
    open(outp, "w", encoding="utf-8").write(srt_content)

    # 1:1 ailrcgenerator 轉換出 LRC 歌詞
    lrc_content = srt_to_lrc(srt_content)
    lrc_path = os.path.splitext(outp)[0] + ".lrc"
    open(lrc_path, "w", encoding="utf-8").write(lrc_content)
    print(f"  [LRC 生成] 已產出 LRC 歌詞: {os.path.basename(lrc_path)}", flush=True)

    chk = os.path.splitext(outp)[0] + "_請檢查.txt"
    if warn:
        open(chk, "w", encoding="utf-8").write("\n".join(warn))
        print(f"  建議檢查 {len(warn)} 句：\n" + "\n".join(warn))
    elif os.path.exists(chk):
        os.remove(chk)
    print(f"  done {time.time()-t0:.0f}s -> {outp}")

    # 最後輸出完的歌曲與歌詞（.mp3, .txt, .srt, .lrc）存入 songs 資料夾
    songs_dir = Path(__file__).resolve().parent / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)
    files_to_sync = [audio, lyr, outp, lrc_path]
    if os.path.exists(chk):
        files_to_sync.append(chk)

    for src_file in files_to_sync:
        if src_file and os.path.exists(src_file):
            target_file = songs_dir / os.path.basename(src_file)
            if target_file.resolve() != Path(src_file).resolve():
                shutil.copy2(src_file, target_file)

    print(f"  [歸檔完成] 歌曲與歌詞檔（音訊/.txt/.srt/.lrc）已存入: {songs_dir}", flush=True)


def run_transcription(audio, outp, lang=None):
    """純聽寫模式：當無歌詞時，使用 Demucs 人聲分離 + Whisper 自動語音辨識 -> 產生 SRT + LRC + TXT。"""
    t0 = time.time()
    base = os.path.splitext(audio)[0]
    txt_path = base + ".txt"
    lrc_path = base + ".lrc"

    print(f"{os.path.basename(audio)}  [純聽寫模式：未附歌詞] ({DEV})", flush=True)
    print("  loading audio...", flush=True)
    x, sr = load_audio(audio)

    print("  separating vocals (Demucs)...", flush=True)
    v16 = separate_vocals(x, sr)
    print(f"  vocals separated  {time.time()-t0:.0f}s", flush=True)

    print("  transcribing speech (Whisper turbo)...", flush=True)
    model = get_whisper()
    torch.manual_seed(0)
    result = model.transcribe(
        v16,
        language=lang,
        word_timestamps=True,
        verbose=None,
        fp16=(DEV == "cuda"),
        condition_on_previous_text=False
    )
    _models.pop("whisper", None)
    import gc
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()

    # 針對歌詞斷句優化：依據停頓與標點分行
    try:
        result.split_by_gap(0.8)
        result.split_by_punctuation([',', '.', '!', '?', '，', '。', '！', '？', '…'])
    except Exception:
        pass

    srt_entries = []
    lyric_lines = []

    for seg in result.segments:
        text = seg.text.strip()
        text = re.sub(r'^[，。！？,.!?…\s]+|[，。！？,.!?…\s]+$', '', text).strip()
        if not text:
            continue
        srt_entries.append(f"{len(srt_entries)+1}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{text}\n")
        lyric_lines.append(text)

    if not srt_entries:
        raise RuntimeError("語音聽寫未辨識出有效文字（可能是純音樂或伴奏）。")

    # 1. 寫入 SRT 字幕檔
    srt_content = "\n".join(srt_entries)
    open(outp, "w", encoding="utf-8").write(srt_content)
    print(f"  [SRT 產出] {os.path.basename(outp)}", flush=True)

    # 2. 備份出純歌詞 TXT 檔
    open(txt_path, "w", encoding="utf-8").write("\n".join(lyric_lines) + "\n")
    print(f"  [歌詞備份] 已自動存出聽寫文字: {os.path.basename(txt_path)}（共 {len(lyric_lines)} 行）", flush=True)

    # 3. 產生 LRC 動態歌詞檔 (1:1 逆向 ailrcgenerator 演算法)
    lrc_content = srt_to_lrc(srt_content)
    open(lrc_path, "w", encoding="utf-8").write(lrc_content)
    print(f"  [LRC 生成] 已產出 LRC 歌詞: {os.path.basename(lrc_path)}", flush=True)

    print(f"  done {time.time()-t0:.0f}s -> {outp}", flush=True)

    # 4. 同步歸檔至 songs/ 目錄
    songs_dir = Path(__file__).resolve().parent / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)
    files_to_sync = [audio, txt_path, outp, lrc_path]
    for src_file in files_to_sync:
        if src_file and os.path.exists(src_file):
            target_file = songs_dir / os.path.basename(src_file)
            if target_file.resolve() != Path(src_file).resolve():
                shutil.copy2(src_file, target_file)

    print(f"  [歸檔完成] 歌曲與所有產物（音訊/.txt/.srt/.lrc）已存入: {songs_dir}", flush=True)


AUDIO_ALL = AUDIO_EXT + (".mp4", ".mov", ".mkv", ".aac", ".wma")


def find_pairs(paths):
    """anything dropped (audio, txt, folder, several at once) -> list of (audio, txt) pairs"""
    bases = []
    for p in paths:
        if os.path.isdir(p):
            bases += [os.path.join(p, os.path.splitext(f)[0]) for f in sorted(os.listdir(p))]
        else:
            bases.append(os.path.splitext(p)[0])
    pairs, seen = [], set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        audio = next((base + e for e in AUDIO_ALL if os.path.exists(base + e)), None)
        txt_path = base + ".txt"

        if audio:
            if os.path.exists(txt_path):
                pairs.append((audio, txt_path))
            else:
                # 嘗試提取音訊內嵌歌詞（Suno/Udio/ID3 tags）
                lines = extract_embedded_lyrics(audio)
                if lines:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    print(f"  [內嵌歌詞] 已從 {os.path.basename(audio)} 提取並自動備份至 {os.path.basename(txt_path)}（共 {len(lines)} 行）", flush=True)
                    pairs.append((audio, txt_path))
                else:
                    print(f"  [純聽寫模式] {os.path.basename(audio)}：未附歌詞且無內嵌標籤，將啟用 GPU 人聲分離 + Whisper 純聽寫模式...", flush=True)
                    pairs.append((audio, None))
        elif os.path.exists(txt_path):
            print(f"  !! {os.path.basename(base)}：缺同名的音訊檔（音訊與歌詞檔名要一致）", flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--lang")
    a = ap.parse_args()

    # 1. 支援直接拖曳 .srt 轉換為 .lrc
    srt_inputs = [p for p in a.paths if os.path.isfile(p) and p.lower().endswith(".srt")]
    non_srt_paths = [p for p in a.paths if not (os.path.isfile(p) and p.lower().endswith(".srt"))]
    for srt_path in srt_inputs:
        print(f"偵測到字幕檔: {os.path.basename(srt_path)}，直接進行 SRT -> LRC 轉換...", flush=True)
        lrc_text = srt_to_lrc(open(srt_path, encoding="utf-8-sig").read())
        lrc_path = os.path.splitext(srt_path)[0] + ".lrc"
        open(lrc_path, "w", encoding="utf-8").write(lrc_text)
        print(f"  [LRC 生成] 已產出 LRC: {os.path.basename(lrc_path)}", flush=True)
        songs_dir = Path(__file__).resolve().parent / "songs"
        songs_dir.mkdir(parents=True, exist_ok=True)
        for f in (srt_path, lrc_path):
            tgt = songs_dir / os.path.basename(f)
            if tgt.resolve() != Path(f).resolve():
                shutil.copy2(f, tgt)
        print(f"  [歸檔完成] 字幕與 LRC 歌詞檔已同步歸檔至: {songs_dir}", flush=True)

    if non_srt_paths:
        pairs = find_pairs(non_srt_paths)
        if not pairs and not srt_inputs:
            print("沒找到可以對齊的歌。請把「歌曲檔」或「歌詞 txt」或「字幕 srt」或「整個資料夾」拖到 DROP_HERE.bat 上。")
        failed = not bool(pairs)
        for audio, txt in pairs:
            try:
                outp = os.path.splitext(audio)[0] + ".srt"
                if txt:
                    run(audio, txt, outp, a.lang)
                else:
                    run_transcription(audio, outp, a.lang)
            except Exception as e:
                failed = True
                print(f"  !! {os.path.basename(audio)} 失敗：{e}")
                traceback.print_exc()
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
