#!/usr/bin/env python3
"""
Patch app/retrieval/embedding.py for the sentence-transformers rename.

`get_sentence_embedding_dimension` became `get_embedding_dimension`. Older
versions only have the former, newer ones warn on it. This makes the lookup
work on both rather than pinning a version.

Idempotent - safe to run twice.

Usage:
    python3 scripts/patch_embedding_dim.py
    python3 scripts/patch_embedding_dim.py --check    # report only
"""

import sys
from pathlib import Path

TARGET = Path("app/retrieval/embedding.py")

OLD = """    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()"""

NEW = '''    @property
    def dim(self) -> int:
        # sentence-transformers renamed this; support both spellings so the
        # pipeline is not pinned to one version of the library.
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:
            getter = model.get_sentence_embedding_dimension
        return getter()'''


def main() -> int:
    check_only = "--check" in sys.argv

    if not TARGET.exists():
        print(f"not found: {TARGET}")
        print("run this from the repo root")
        return 1

    text = TARGET.read_text()

    if NEW in text:
        print("already patched - nothing to do")
        return 0

    if OLD not in text:
        print(f"could not find the expected block in {TARGET}")
        print("the file may have been edited by hand; patch it manually:")
        print()
        print(NEW)
        return 1

    if check_only:
        print(f"{TARGET} needs the patch (use without --check to apply)")
        return 0

    backup = TARGET.with_suffix(".py.bak")
    backup.write_text(text)
    TARGET.write_text(text.replace(OLD, NEW))
    print(f"patched {TARGET}")
    print(f"backup  {backup}")

    # confirm it still parses before declaring success
    import ast
    try:
        ast.parse(TARGET.read_text())
        print("syntax ok")
    except SyntaxError as e:
        TARGET.write_text(backup.read_text())
        print(f"syntax error, reverted: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
