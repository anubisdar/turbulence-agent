#!/usr/bin/env python3
"""
Embed the conclusion-bearing chunks into the vector index.

Separate pass from ingest, so a re-embed never means re-parsing and a schema
change never means re-downloading. Resumable: only chunks with no
`embedded_at` are processed, so an interrupted run picks up where it stopped.

Factual chunks are skipped by design - they are stored and retrievable by
case, not by similarity. See app/retrieval/chunking.EMBEDDED_SECTIONS.

Usage:
    python3 scripts/embed_chunks.py --limit 50     # smoke test first
    python3 scripts/embed_chunks.py
    python3 scripts/embed_chunks.py --reembed      # discard and redo
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.chunking import EMBEDDED_SECTIONS  # noqa: E402
from app.retrieval.embedding import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    SentenceTransformerEncoder,
    batched,
    serialize,
)
from app.retrieval.schema import connect, init_db  # noqa: E402

DEFAULT_DB = "data/retrieval.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N chunks (smoke test)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--reembed", action="store_true",
                    help="clear existing vectors and start over")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"No database at {db_path}. Run ingest_ntsb.py first.")

    print("loading encoder (first run downloads the model)...")
    encoder = SentenceTransformerEncoder(batch_size=args.batch_size)
    dim = encoder.dim
    print(f"  {encoder.name}  dim={dim}")

    conn = connect(db_path, load_vec=True)
    init_db(conn, embedding_dim=dim)

    if args.reembed:
        conn.execute("DELETE FROM chunk_vec")
        conn.execute("UPDATE chunks SET embedded_at = NULL, embedding_model = NULL")
        conn.commit()
        print("cleared existing vectors")

    sections = ",".join("?" for _ in EMBEDDED_SECTIONS)
    params = [s.value for s in EMBEDDED_SECTIONS]

    total_pending = conn.execute(
        f"SELECT COUNT(*) FROM chunks WHERE embedded_at IS NULL "
        f"AND section IN ({sections})", params
    ).fetchone()[0]
    skipped = conn.execute(
        f"SELECT COUNT(*) FROM chunks WHERE section NOT IN ({sections})", params
    ).fetchone()[0]

    print(f"\nchunks to embed:      {total_pending:,}")
    print(f"stored, not embedded: {skipped:,}  (factual narratives)")
    if not total_pending:
        print("nothing to do")
        return

    rows = conn.execute(
        f"""SELECT id, context_header, text FROM chunks
            WHERE embedded_at IS NULL AND section IN ({sections})
            ORDER BY section_priority, id
            {'LIMIT ' + str(args.limit) if args.limit else ''}""",
        params,
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    done = 0
    started = time.time()

    for batch in batched(rows, args.batch_size):
        texts = [f"{r['context_header']}\n\n{r['text']}" for r in batch]
        vectors = encoder.encode(texts)
        for row, vec in zip(batch, vectors):
            conn.execute(
                "INSERT OR REPLACE INTO chunk_vec(chunk_id, embedding) VALUES (?, ?)",
                (row["id"], serialize(vec)),
            )
            conn.execute(
                "UPDATE chunks SET embedded_at = ?, embedding_model = ? WHERE id = ?",
                (now, encoder.name, row["id"]),
            )
        done += len(batch)
        if done % (args.batch_size * 10) == 0 or done == len(rows):
            conn.commit()
            rate = done / max(time.time() - started, 0.001)
            remaining = (len(rows) - done) / max(rate, 0.001)
            print(f"  {done:,}/{len(rows):,}  {rate:.1f}/s  "
                  f"~{remaining/60:.1f} min left")
    conn.commit()

    print(f"\nembedded {done:,} chunks in {(time.time()-started)/60:.1f} min")

    vec_count = conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL"
    ).fetchone()[0]
    print(f"vectors in index:  {vec_count:,}")
    print(f"chunks marked:     {embedded:,}")
    if vec_count != embedded:
        print("  WARNING: counts disagree - vectors and metadata are out of sync")

    print(f"database size: {db_path.stat().st_size/1024/1024:.1f} MB")
    conn.close()


if __name__ == "__main__":
    main()
