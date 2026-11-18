"""Small helpers for the quality gates: a check table that fails loudly, and a degenerate-column check."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


class Gate:
    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), str(detail)))
        return bool(ok)

    def table(self) -> str:
        w = max(len(r[0]) for r in self.rows) if self.rows else 10
        lines = [f"\nGATE  {self.title}", f"  {'check':{w}s}  result  detail"]
        for name, ok, detail in self.rows:
            lines.append(f"  {name:{w}s}  {'PASS' if ok else 'FAIL':6s}  {detail}")
        failed = [r for r in self.rows if not r[1]]
        lines.append(f"GATE RESULT: {'PASS' if not failed else 'FAIL'} "
                     f"({len(self.rows) - len(failed)}/{len(self.rows)} checks passed)")
        return "\n".join(lines)

    def finish(self, exit_on_fail: bool = True) -> bool:
        print(self.table(), flush=True)
        ok = all(r[1] for r in self.rows)
        if not ok and exit_on_fail:
            sys.exit(1)
        return ok


def describe_numeric(s: pd.Series, may_be_constant: bool = False, may_be_zero: bool = False) -> dict:
    """min / mean / max / distinct / non-finite count, and whether the column is degenerate."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    nonfinite = int((~np.isfinite(x.to_numpy())).sum())
    fin = x[np.isfinite(x)]
    nd = int(fin.nunique())
    allzero = bool(len(fin) and (fin == 0).all())
    ok = nonfinite == 0 and len(fin) > 0 and (may_be_constant or nd > 1) and (may_be_zero or not allzero)
    text = (f"min {fin.min():.6g}  mean {fin.mean():.6g}  max {fin.max():.6g}  distinct {nd}  non-finite {nonfinite}"
            if len(fin) else f"no finite values (non-finite {nonfinite})")
    return {"ok": ok, "text": text, "min": float(fin.min()) if len(fin) else None,
            "mean": float(fin.mean()) if len(fin) else None, "max": float(fin.max()) if len(fin) else None,
            "distinct": nd, "nonfinite": nonfinite}


def degenerate_table(g: Gate, df: pd.DataFrame, label: str, constant_ok: tuple = (), zero_ok: tuple = ()) -> None:
    """Add one degenerate-column check per numeric column of df to the gate."""
    for col in df.columns:
        if df[col].dtype == bool:
            vals = df[col].astype(int)
        elif pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col]
        else:
            continue
        d = describe_numeric(vals, may_be_constant=col in constant_ok, may_be_zero=col in zero_ok)
        g.check(f"{label}.{col}", d["ok"], d["text"])
