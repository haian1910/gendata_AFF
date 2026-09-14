"""Sync corpus D exactly as an eval pod does (evalsrv.corpus.CorpusSync:
manifest pointer -> immutable copy check -> every active chunk + the parquet
index, sha256-verified). Default is the live manifest (latest corpus_epoch);
pass --manifest-key corpus/manifests/<sha>.json to replay a pinned revision.

gen.py never refreshes: it uses whatever this script last verified, so a
datagen run stays on one epoch even if the live corpus moves underneath.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tomllib
from pathlib import Path

from affine import dialects
from evalsrv.corpus import CorpusSync

TG_ROOT = Path(os.environ.get("TG_ROOT", "/dev/shm/affine-teachergen"))


def main() -> None:
    contract = tomllib.loads(dialects.CONTRACT_TOML.read_text())["dataset"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-key", default=contract["manifest_key"])
    ap.add_argument("--base-url", default=contract["corpus_base_url"])
    ap.add_argument("--data-dir", type=Path, default=TG_ROOT / "corpus")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    sync = CorpusSync(args.base_url, args.manifest_key, args.data_dir)
    if not sync.refresh() or sync.stale:
        sys.exit(f"corpus sync failed (ready={sync.ready}, stale={sync.stale})")
    print(json.dumps(sync.info(), indent=2))


if __name__ == "__main__":
    main()
