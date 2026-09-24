"""Build the aa_c1 prompt set: ~10K tokens of REAL PROSE plus an instruction targeting >=1500 output.

WHY THIS FILE EXISTS. aa_c1 mirrors Artificial Analysis's ~10K-in / 1500-out single-stream cell. It
must use real prose, not random tokens: the MTP draft model can predict the continuation of natural
language and cannot predict random tokens, so acceptance falls from ~2.71 to ~2.00 and the cell
under-reports by about 25% (measured: 283.8 tok/s on random tokens vs 371.7 on prose). The shape is
the same; the number is not.

SOURCE. Public-domain text. The reference set used Dickens, "A Tale of Two Cities" (Project
Gutenberg #98). Any sufficiently long prose works -- absolute tok/s will differ slightly with the
passage, but acceptance lands in the right regime, which is the point.

USAGE (inside the serving container, which already has the tokenizer):
    python3 make_aa_prompts.py --out /tmp/aa.jsonl                 # downloads #98
    python3 make_aa_prompts.py --src book.txt --out /tmp/aa.jsonl  # offline

Output is one {"prompt": ...} per line, the format `vllm bench serve --dataset-name custom` wants.
"""
import argparse, json, sys

GUTENBERG = "https://www.gutenberg.org/files/98/98-0.txt"

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

    if a.src:
        text = open(a.src, encoding="utf-8", errors="ignore").read()
    else:
        import urllib.request
        print(f"downloading {GUTENBERG}", file=sys.stderr)
        text = urllib.request.urlopen(GUTENBERG, timeout=60).read().decode("utf-8", "ignore")

    # Strip Gutenberg boilerplate so the prompt is prose, not a licence header.
    for mark in ("*** START OF TH", "*** START OF THE PROJECT"):
        i = text.find(mark)
        if i != -1:
            text = text[text.find("\n", i) + 1:]
            break
    j = text.find("*** END OF TH")
    if j != -1:
        text = text[:j]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    instr_n = len(tok(INSTRUCTION).input_ids)
    budget = a.target_in - instr_n

    ids = tok(text).input_ids
    if len(ids) < budget * a.n:
        print(f"WARNING: source has {len(ids)} tokens, need ~{budget*a.n}; passages will overlap",
              file=sys.stderr)

    rows, lens = [], []
    for k in range(a.n):
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
