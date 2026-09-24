# aa_prompts.jsonl — the exact prompt set behind the `aa_c1` number

40 prompts, ~10,000 Gemma tokens each, one `{"prompt": ...}` per line. This is the set the
reference table's **371.7 tok/s** was measured with, shipped verbatim so the cell is reproducible
rather than approximated.

## Why the exact prompts matter

`aa_c1` is single-stream with MTP n=3 speculative decoding, so throughput scales close to linearly
with draft acceptance — and acceptance depends on how predictable the prose is. Measured on the
same machine and config, varying only the prompts:

| corpus | acceptance | tok/s |
| --- | ---: | ---: |
| random tokens | 2.00 | 283.8 |
| A Tale of Two Cities alone | 2.494 | 343.1 |
| the AA-harness set (prose + a 1,500-word instruction) | 2.43 | 343.1 |
| **this set** | **2.713** | **371.7** |

Verified: run from a second machine over an SSH tunnel, this set gives **369.48 tok/s**, −0.6% from
the reference. The three wrong corpora above all land near 343 — which is what made the cell look
like a machine difference when it was a data-packaging one.

A 31% spread in the headline number from the prompt text alone. That is why the cell ships its data.

## Contents

A mix of public-domain works: **Moby-Dick** (Project Gutenberg #2701, ~10 passages),
**A Tale of Two Cities** (#98, ~4) and **Frankenstein** (#84, ~4), each truncated to the token
budget and followed by an instruction asking for a ≥1,500-word literary analysis (AA does not use
`--ignore-eos`, so output length has to come from the instruction).

## A known flaw, stated rather than hidden

**Some prompts contain Project Gutenberg licence boilerplate**, which
the original generator did not strip. Boilerplate is formulaic and a draft model predicts it easily,
so those prompts inflate acceptance. **371.7 is therefore slightly optimistic** — about 12.5% of it
was measured on text nobody would serve.

`make_aa_prompts.py` strips the boilerplate, so a set built with it is cleaner but will score
slightly lower. Use this file to reproduce the published number; use the generator for an honest
measurement of your own.

The underlying works are public domain. Project Gutenberg asks that its trademark and licence header
not be attached to redistributed text — the three affected prompts contain fragments of it, which is
the flaw noted above and a further reason to prefer the generator for new work.
