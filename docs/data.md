# Preparing USPTO-ORD-100K

> One-time dataset setup for the machine that will run training.
> Produces the Alpaca-format JSON files read by
> `hierarchical_soft_prompt.data.load_uspto_ord`.
> No `ord_schema` / protobuf install required — we use the prebuilt 7z.

---

## Context

USPTO-ORD-100K is not on HuggingFace. It is published via
[`qai222/LLM_organic_synthesis`](https://github.com/qai222/LLM_organic_synthesis) from
the paper *Extracting structured data from organic synthesis procedures* (RSC Digital
Discovery 2024, `10.1039/D4DD00091A`).

The repo ships a prebuilt 100K subset as a committed 7z archive at
`workplace_data/datasets/USPTO-n100k-t2048_exp1.7z`. Extract it once, point the loader
at the extracted directory, done.

Deterministic 80/10/10 split (seed 42) inside the archive:
- `train.json` — 80,000 reactions
- `valid.json` — 10,000 reactions
- `test.json`  — 10,000 reactions

Each file is a JSON list of `{"instruction": str, "output": str}` in Alpaca format
(see § Sample format below).

License: CC-BY-SA 4.0 (share-alike — derivatives must match).

---

## Steps

```bash
# 1. Pick a location on the server with ≥ 500 MB free.
#    Same filesystem as the training code is fine; outside the git tree is cleaner.
export USPTO_DIR=$HOME/datasets/uspto_ord
mkdir -p "$USPTO_DIR" && cd "$USPTO_DIR"

# 2. Shallow-clone only the data we need (avoids ~500 MB of Python code + demos).
#    --filter=blob:none + sparse-checkout keeps the clone under 50 MB.
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/qai222/LLM_organic_synthesis.git
cd LLM_organic_synthesis
git sparse-checkout set workplace_data/datasets

# 3. Install 7z if not already present.
#    Ubuntu/Debian: sudo apt install -y p7zip-full
#    CentOS/RHEL:   sudo yum install -y p7zip
#    (On Windows WSL, use the same apt command inside the distro.)

# 4. Extract the prebuilt subset.
cd workplace_data/datasets
7z x USPTO-n100k-t2048_exp1.7z
# → creates ./USPTO-n100k-t2048_exp1/{train,valid,test,meta}.json

# 5. Sanity check.
python -c "
import json, os
d = 'USPTO-n100k-t2048_exp1'
for split in ('train', 'valid', 'test'):
    with open(os.path.join(d, f'{split}.json')) as f:
        rows = json.load(f)
    print(f'{split}: {len(rows)} rows')
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert {'instruction', 'output'} <= set(rows[0].keys())
print('OK')
"
# Expected:
#   train: 80000 rows
#   valid: 10000 rows
#   test:  10000 rows
#   OK

# 6. Expose the path via env var (or pass --data_dir at call time).
#    Add this to ~/.bashrc or the job-launch wrapper.
export USPTO_ORD_DATA=$USPTO_DIR/LLM_organic_synthesis/workplace_data/datasets/USPTO-n100k-t2048_exp1
```

The loader reads from the directory pointed to by `$USPTO_ORD_DATA`, or from the
path passed via `--data_dir`.

---

## Sample format

Each row in each split:

```json
{
  "instruction": "Below is a description of an organic reaction. Extract information from it to an ORD JSON record.\n\n### Procedure:\n<procedure text here>\n\n### ORD JSON:\n",
  "output": "{\"inputs\": {...}, \"conditions\": {...}, \"outcomes\": [...], \"workups\": [...]}"
}
```

- `instruction` wraps the procedure paragraph in the repo's default template. Our
  loader recovers raw procedure text by slicing between `### Procedure:\n` and
  `\n\n### ORD JSON:`.
- `output` is a **JSON string** (not a dict). Produced via
  `json.dumps(MessageToDict(reaction, preserving_proto_field_name=True))`. Top-level
  keys are a subset of `{inputs, conditions, outcomes, workups}`. Depth up to ~7
  levels in places (e.g. `inputs.<key>.components[].amount.mass.value`).

---

## If you want to rebuild from scratch (not needed for this project)

The 7z above is sufficient for all our experiments. For reproducibility / paper-audit
purposes only:

1. Clone raw ORD data (~10 GB): `git clone https://github.com/open-reaction-database/ord-data`.
2. Install: `pip install ord-schema protobuf rdkit-pypi loguru pydantic tqdm pandas`.
3. Edit `LOCAL_DATA_FOLDER` in `workplace_data/uspto/export_from_pb.py` to point at the
   ord-data clone.
4. Run `python workplace_data/uspto/export_from_pb.py` → produces `export_from_pb_dedup.json.gz`.
5. Run `python workplace_data/prepare_uspto.py` → rebuilds the 80/10/10 split.

Skip unless specifically asked. The committed 7z is produced by exactly this pipeline.

---

## Troubleshooting

- **`7z: command not found`** — install `p7zip-full` (Ubuntu) or `p7zip` (RHEL).
- **`git-lfs required`** — the 7z is a committed blob, not LFS; this error means the
  clone is truncated. Retry `git sparse-checkout set workplace_data/datasets` inside
  the repo.
- **Extracted directory empty** — check `7z x` exit code. The archive is single-part;
  `7z x` (not `7z e`) preserves the nested `USPTO-n100k-t2048_exp1/` folder.
- **Loader reports 0 samples** — double-check `$USPTO_ORD_DATA` points at the
  directory *containing* `train.json`, not its parent.
