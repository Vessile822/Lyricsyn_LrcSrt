"""Boundary refinement with MMS (multilingual CTC, 1000+ languages via uroman romanization).
Whisper decides WHICH stretch of audio a line is; MMS then decides exactly where its first and last sounds are,
inside a small window. Lines where MMS is unsure keep the Whisper times."""

import re

import numpy as np
import torch

CJK = r"[぀-ヿ㐀-鿿가-힯]"

MMS = "MahmoudAshraf/mms-300m-1130-forced-aligner"
_m = {}
DEBUG = False  # print per-word MMS times
HUM_FIX = True
HUM_GAP = 0.8
GLUE = (0.6, 2.5)
HUM_STRICT = False


def _load(dev):
    if "model" not in _m:
        import uroman as ur
        from transformers import AutoModelForCTC, AutoTokenizer

        _m["tok"] = AutoTokenizer.from_pretrained(MMS).get_vocab()
        _m["model"] = (
            AutoModelForCTC.from_pretrained(MMS, dtype=torch.float32)
            .to(dev)
            .eval()
        )  # fp16 -> NaN on GTX16
        _m["ur"] = ur.Uroman()
    return _m["model"], _m["tok"], _m["ur"]


def _emissions(model, x, dev):
    x = torch.from_numpy(x)
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.no_grad():
        lg = model(x[None].to(dev)).logits[0].float()
    return torch.log_softmax(lg, -1).cpu().numpy()  # 20 ms frames


def _viterbi(lp, seq, blank):
    """CTC forced alignment -> per-frame state index (odd = token seq[(s-1)//2])"""
    T = lp.shape[0]
    ext = [blank]
    for t in seq:
        ext += [t, blank]
    ext = np.array(ext)
    S = len(ext)
    skip = np.zeros(S, bool)
    for s in range(2, S):
        skip[s] = ext[s] != blank and ext[s] != ext[s - 2]
    NEG = -1e30
    dp = np.full(S, NEG)
    dp[0] = lp[0, ext[0]]
    dp[1] = lp[0, ext[1]]
    bp = np.zeros((T, S), np.int8)
    em = lp[:, ext]
    for t in range(1, T):
        b = np.concatenate(([NEG], dp[:-1]))
        c = np.where(skip, np.concatenate(([NEG, NEG], dp[:-2])), NEG)
        st = np.stack([dp, b, c])
        k = st.argmax(0)
        dp = st[k, np.arange(S)] + em[t]
        bp[t] = k
    s = S - 1 if dp[S - 1] > dp[S - 2] else S - 2
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return ext, path


def refine(v16, spans, lines, dev, pad=1.0, max_move=1.0, min_conf=0.15, star_pen=-2.0):
    model, vocab, U = _load(dev)
    blank = vocab.get("<blank>", 0)
    out = []
    for i, (st, en, p) in enumerate(spans):
        prev_end = out[-1][1] if out else 0.0
        nxt = spans[i + 1][0] if i + 1 < len(spans) else len(v16) / 16000
        w0 = max(prev_end, st - pad)
        w1 = min(nxt, en + 0.5)
        seq, unit = [], []  # token ids + which unit (CJK char / word) each belongs to
        for u, piece in enumerate(re.findall(CJK + r"|[^\s぀-ヿ㐀-鿿가-힯]+", lines[i])):
            for c in U.romanize_string(piece).lower():
                if c in vocab and c != "|":
                    seq.append(vocab[c])
                    unit.append(u)
        if w1 - w0 < 0.4 or len(seq) < 2:
            out.append((st, en, p))
            continue
        lp = _emissions(model, v16[int(w0 * 16000) : int(w1 * 16000)], dev)
        star = lp.shape[1]
        lp = np.concatenate(
            [lp, lp[:, 4:].max(1, keepdims=True) + star_pen], 1
        )  # wildcard absorbs ad-libs / neighbours
        ext, path = _viterbi(lp, [star] + seq + [star], blank)
        tok = [(t, s) for t, s in enumerate(path) if s % 2 == 1 and ext[s] != star]
        if not tok:
            out.append((st, en, p))
            continue
        conf = float(np.exp(np.mean([lp[t, ext[s]] for t, s in tok])))
        nst, nen = w0 + tok[0][0] * 0.02, w0 + (tok[-1][0] + 1) * 0.02
        # humming before the first word gets glued onto it: first unit far longer than the rest -> start later
        ust = {}
        for t, s_ in tok:
            ust.setdefault(
                unit[(s_ - 1) // 2 - 1], w0 + t * 0.02
            )  # -1: leading star token
        us = [ust[k] for k in sorted(ust)]
        if len(us) >= 4:
            med = float(np.median(np.diff(us[1:])))
            if us[1] - us[0] > max(GLUE[0], GLUE[1] * med):
                nst = us[1] - max(med, 0.15)
            if DEBUG:
                print(
                    "units",
                    lines[i][:6],
                    [round(x, 2) for x in us],
                    "conf",
                    round(conf, 3),
                    "st",
                    st,
                )
        hum = nst - st > HUM_GAP and (
            not HUM_STRICT or len(us) < 4 or us[1] - us[0] < max(0.8, 3 * med)
        )  # whisper started on humming; MMS heard the first real word later
        if (
            conf >= min_conf and abs(nst - st) <= max_move or hum and HUM_FIX
        ) and nen > nst:
            out.append((nst, en, p))  # ends are handled by extend_tails
        else:
            out.append((st, en, p))
    return out
