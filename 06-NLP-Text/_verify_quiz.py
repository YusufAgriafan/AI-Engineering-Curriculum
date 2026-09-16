"""Verify quiz notebook: inject perfect answers, run grader, expect full score.

Usage (dari root repo):
    python AI-Engineering-Curriculum/06-NLP-Text/_verify_quiz.py
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

NB = Path("AI-Engineering-Curriculum/06-NLP-Text/02_kuis_nlp_text.ipynb")
EXPECTED = 23
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ANSWERS = {
    "JAWABAN": {1: "a", 2: "b", 3: "c", 4: "c"},
    "pad_sequences": (
        "def pad_sequences(seqs, maxlen):\n"
        "    out = np.full((len(seqs), maxlen), PAD_ID, dtype=int)\n"
        "    for i, s in enumerate(seqs):\n"
        "        potong = s[:maxlen]\n"
        "        out[i, :len(potong)] = potong\n"
        "    return out\n"
    ),
    "pasangan_skipgram": (
        "def pasangan_skipgram(kalimat_list, w2i, window=1):\n"
        "    pairs = []\n"
        "    for toks in kalimat_list:\n"
        "        ids = [w2i[t] for t in toks]\n"
        "        for i, t in enumerate(ids):\n"
        "            for j in range(max(0, i - window), min(len(ids), i + window + 1)):\n"
        "                if j != i:\n"
        "                    pairs.append((t, ids[j]))\n"
        "    return pairs\n"
    ),
    "sg_step": (
        "def sg_step(v, u, lr):\n"
        "    v = np.asarray(v, dtype=float).copy()\n"
        "    u = np.asarray(u, dtype=float).copy()\n"
        "    s = sigmoid(float(v @ u))\n"
        "    loss = -np.log(max(s, 1e-12))\n"
        "    grad_v = (s - 1.0) * u\n"
        "    grad_u = (s - 1.0) * v\n"
        "    v -= lr * grad_v\n"
        "    u -= lr * grad_u\n"
        "    return (loss, v, u)\n"
    ),
    "bigram_matrix": (
        "def bigram_matrix(seq, vocab_size, alpha=0.5):\n"
        "    counts = np.full((vocab_size, vocab_size), 1.0 + alpha)\n"
        "    for a, b in zip(seq[:-1], seq[1:]):\n"
        "        counts[a, b] += 1.0\n"
        "    return counts / counts.sum(axis=1, keepdims=True)\n"
    ),
    "sample_with_temp": (
        "def sample_with_temp(logits, temp, rng):\n"
        "    logits = np.asarray(logits, dtype=float)\n"
        "    z = logits - logits.max()\n"
        "    p = np.exp(z / temp)\n"
        "    p = p / p.sum()\n"
        "    return int(rng.choice(len(logits), p=p))\n"
    ),
    "rnn_step_forward": (
        "def rnn_step_forward(x_t, h_prev, Wx, Wh, bh):\n"
        "    h_pre = x_t @ Wx + h_prev @ Wh.T + bh\n"
        "    h = np.tanh(h_pre)\n"
        "    return (h, (h, h_pre))\n"
    ),
    "rnn_step_backward": (
        "def rnn_step_backward(dh, cache):\n"
        "    h, h_pre = cache\n"
        "    return dh * (1 - np.tanh(h_pre) ** 2)\n"
    ),
}

ctx = {"__name__": "__main__"}
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "JAWABAN = {" in src and "None" in src:
        exec("JAWABAN = " + repr(ANSWERS["JAWABAN"]), ctx)
        print(f"[cell {i}] injected JAWABAN")
        continue
    if "TODO" in src:
        names = [n for n in ANSWERS if n != "JAWABAN" and f"def {n}(" in src]
        if not names:
            print(f"[cell {i}] SKIPPED (TODO tanpa fungsi dikenal)")
            continue
        # jalankan bagian cell yang diberikan (impor, konstanta, fungsi bantu),
        # tapi lewati seluruh blok fungsi yang diganti implementasi referensi
        keep = []
        skipping = False
        skip_indent = 0
        for ln in src.splitlines():
            if not skipping:
                m = re.match(r"def (\w+)\(", ln.strip())
                if m and m.group(1) in names:
                    skipping = True
                    skip_indent = len(ln) - len(ln.lstrip())
                    continue
                keep.append(ln)
            else:
                if ln.strip() == "" or (len(ln) - len(ln.lstrip())) > skip_indent:
                    continue
                skipping = False
                keep.append(ln)
        pre_src = "\n".join(keep)
        if pre_src.strip():
            exec(compile(pre_src, f"<cell {i} pre>", "exec"), ctx)
        for n in names:
            exec(ANSWERS[n], ctx)
        print(f"[cell {i}] injected: {names}")
        continue
    if "jalankan_kuis()" in src:
        # jalankan grader dengan stdout ditangkap supaya skor bisa di-parse
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"<cell {i}>", "exec"), ctx)
        out = buf.getvalue()
        print(out.rstrip())
        for ln in out.splitlines():
            if "SKOR AKHIR" in ln:
                skor = int(ln.split(":")[1].strip().split("/")[0])
                if skor != EXPECTED:
                    print(f"FAILED: skor {skor} != {EXPECTED}")
                    sys.exit(1)
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ctx)
    print(f"[cell {i}] ok")

print(f"\nQUIZ VERIFY OK: skor penuh {EXPECTED}/{EXPECTED}")
