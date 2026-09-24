"""Build the aa_c1 prompt set: ~10K tokens of REAL PROSE plus an instruction targeting >=1500 output.

WHY THIS FILE EXISTS. aa_c1 mirrors Artificial Analysis's ~10K-in / 1500-out single-stream cell. It
must use real prose, not random tokens: the MTP draft model can predict the continuation of natural
language and cannot predict random tokens, so acceptance falls from ~2.71 to ~2.00 and the cell
under-reports by about 25% (measured: 283.8 tok/s on random tokens vs 371.7 on prose). The shape is
the same; the number is not.

SOURCE. The reference set is a MIX of three public-domain books, not one:
    Moby-Dick (#2701)  x10    A Tale of Two Cities (#98)  x4    Frankenstein (#84)  x4
plus 3 prompts that unintentionally contained Project Gutenberg LICENCE BOILERPLATE and 3
unidentified. The mix matters: a single book gives different draft acceptance, and acceptance sets
the number. Measured on Dickens alone: 2.494 and 343.1 tok/s; the reference mix: 2.713 and 371.7.

CAVEAT ON THE REFERENCE. Those 3 boilerplate prompts (12.5% of the set) are formulaic and highly
predictable, so they inflate acceptance. This generator STRIPS the boilerplate, which is correct but
means a clean run may land slightly below 371.7 through no fault of the machine.

USAGE (inside the serving container, which already has the tokenizer):
    python3 make_aa_prompts.py --out /tmp/aa.jsonl                 # downloads #98
    python3 make_aa_prompts.py --src book.txt --out /tmp/aa.jsonl  # offline

Output is one {"prompt": ...} per line, the format `vllm bench serve --dataset-name custom` wants.
"""
import argparse, json, sys

# (gutenberg id, weight) — weights reproduce the reference mix
SOURCES = [(2701, 10), (98, 4), (84, 4)]
URLS = ["https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt",
        "https://www.gutenberg.org/files/{i}/{i}-0.txt"]

INSTRUCTION = (
    "\n\n---\n\n"
    "Write a thorough literary analysis of the passage above. Cover, in this order and at length: "
    "(1) the narrative structure and how the scene is staged; "
    "(2) each character who speaks or is described, and what the text reveals about their motives; "
    "(3) the historical and political setting the passage assumes, and how it shapes the events; "
    "(4) the author's prose style, with specific quoted examples and close reading of at least six of them; "
    "(5) the major themes, and how they are developed through imagery and diction; "
    "(6) how this passage functions within the larger work. "
    "Develop each section fully in multiple paragraphs of continuous prose. "
    "Your response must be at least 1,500 words long."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/aa.jsonl")
    ap.add_argument("--src", help="local UTF-8 text file; omit to download public-domain source")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--revision", default="4d7ae4984b7db7de8f8457170b3f1a419ee76d52")
    ap.add_argument("-n", type=int, default=24, help="number of prompts")
    ap.add_argument("--target-in", type=int, default=10000, help="input tokens per prompt")
    a = ap.parse_args()

    def strip_boilerplate(t):
        """Remove the Gutenberg header/footer. The reference set skipped this and 3 of its 24
        prompts carried licence text, which a draft model predicts very easily."""
        i = t.find("*** START OF TH")
        if i != -1:
            t = t[t.find("\n", i) + 1:]
        j = t.find("*** END OF TH")
        if j != -1:
            t = t[:j]
        return t

    if a.src:
        texts = [(strip_boilerplate(open(a.src, encoding="utf-8", errors="ignore").read()), 1)]
    else:
        import urllib.request
        texts = []
        for gid, w in SOURCES:
            raw = None
            for u in URLS:
                try:
                    print(f"downloading gutenberg #{gid}", file=sys.stderr)
                    raw = urllib.request.urlopen(u.format(i=gid), timeout=60).read().decode("utf-8", "ignore")
                    break
                except Exception as e:
                    print(f"  {type(e).__name__}: {e}", file=sys.stderr)
            if raw:
                texts.append((strip_boilerplate(raw), w))
        if not texts:
            sys.exit("could not download any source; use --src with a local text file")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    instr_n = len(tok(INSTRUCTION).input_ids)
    budget = a.target_in - instr_n

    # allocate prompts across sources in the reference proportions
    total_w = sum(w for _, w in texts)
    quota = [max(1, round(a.n * w / total_w)) for _, w in texts]
    while sum(quota) > a.n:
        quota[quota.index(max(quota))] -= 1
    while sum(quota) < a.n:
        quota[quota.index(max(quota))] += 1

    rows, lens = [], []
    for (text, _), want in zip(texts, quota):
        ids = tok(text).input_ids
        if len(ids) < budget * want:
            print(f"WARNING: a source has {len(ids)} tokens, need ~{budget*want}; passages overlap",
                  file=sys.stderr)
        for k in range(want):
            start = (k * budget) % max(1, len(ids) - budget)
            body = tok.decode(ids[start:start + budget], skip_special_tokens=True)
            p = body + INSTRUCTION
            rows.append({"prompt": p})
            lens.append(len(tok(p).input_ids))

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lens.sort()
    print(f"wrote {a.out}: {len(rows)} prompts, "
          f"tokens min {lens[0]} med {lens[len(lens)//2]} max {lens[-1]} (target {a.target_in})")


if __name__ == "__main__":
    main()
