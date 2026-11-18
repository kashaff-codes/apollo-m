"""
Every corpus APOLLO-M has been run on, and where each one's outputs live.

The pipeline writes to one directory per corpus. Until November that rule was
implicit — top-level ``outputs/`` was the benchmark, ``outputs/real/`` the
August API corpus — and nothing enforced it. Merging the Arctic Shift rebuild
(#2) showed the cost: its outputs landed in top-level ``outputs/`` and replaced
the 60-community benchmark that the paper's reproduction scripts, the
dashboard's "Benchmark" banner and ``validate_simulation.py`` all read as the
benchmark. No textual conflict, nothing failed, and every consumer would have
shown real Reddit data under the label "declared simulation".

Naming the datasets here, once, is what stops that recurring: tests, the
dashboard and tools ask this module for a directory instead of each hard-coding
``outputs/``. Adding a corpus means adding one entry below.

``forecast`` records what the rows in that dataset's ``forecast_results.csv``
actually are, because the file layout is identical across datasets while the
meaning is not:

  * ``hindcast``  — ``gen_forecasts.py`` before November: predictions of the
    last five *observed* days, stamped with dates after the data. The values
    are real model output; the dates are not when they apply. Shown against
    the observed values on their true dates, never as the future.
  * ``held_out``  — predictions of observed days kept out of training, with
    the observed value alongside (``actual``). An out-of-sample check.
  * ``future``    — predictions of days after the last observation, with a
    separate backtest of the model on held-out days.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str            # short name for selectors
    kind: str             # "benchmark" | "real"
    path: Path            # directory holding meso_report.csv & co.
    daily: Path           # per-community daily series the forecaster consumed
    source: str
    window: str
    communities: int
    forecast: str         # "hindcast" | "held_out" | "future"
    credit: str
    summary: str          # one sentence for the banner above every chart

    @property
    def is_real(self) -> bool:
        return self.kind == "real"

    def file(self, name: str) -> Path:
        return self.path / name

    def available(self) -> bool:
        return (self.path / "meso_report.csv").exists()

    @property
    def future_file(self) -> Path | None:
        """Forecasts of days after the data, if this dataset has any."""
        own = self.path / "forecast_future.csv"          # gen_forecasts.py, November on
        if own.exists():
            return own
        if self.forecast == "future":                    # the rebuild wrote them as results
            return self.path / "forecast_results.csv"
        return None

    @property
    def backtest_file(self) -> Path | None:
        p = self.path / "forecast_backtest.csv"
        return p if p.exists() else None


DATASETS: dict[str, Dataset] = {
    d.key: d
    for d in [
        Dataset(
            key="benchmark",
            label="Benchmark · declared simulation",
            kind="benchmark",
            path=OUTPUTS,
            daily=ROOT / "data" / "apollo_daily.csv",
            source="simulate_data.py — real toxic-bert scores on planted per-community trajectories",
            window="120 days",
            communities=60,
            forecast="hindcast",
            credit="APOLLO-M team",
            summary=(
                "Declared simulation with ground truth recorded in data/ground_truth.json "
                "and read by no pipeline stage — the only corpus where recall can be "
                "measured, because the answer is known in advance."
            ),
        ),
        Dataset(
            key="real_api_2026_07",
            label="Real · Arctic Shift API · Jul–Aug 2026",
            kind="real",
            path=OUTPUTS / "real",
            daily=OUTPUTS / "real_reddit_daily.csv",
            source="Arctic Shift search API, collected live",
            window="2026-07-02 → 2026-08-11",
            communities=24,
            forecast="hindcast",
            credit="Kashaf Fatima",
            summary=(
                "116,853 genuine comments collected through the Arctic Shift API and "
                "scored by unitary/toxic-bert. The corpus behind the public web view."
            ),
        ),
        Dataset(
            key="real_torrent_2026_08",
            label="Real · RC_2026-08 full monthly dump",
            kind="real",
            path=OUTPUTS / "real_torrent_2026_08",
            daily=OUTPUTS / "real_torrent_2026_08" / "daily_series.csv",
            source="Academic Torrents RC_2026-08.zst (51.5 GB, 367.5M comments), every kept row re-verified",
            window="2026-08-01 → 2026-08-31",
            communities=39,
            forecast="held_out",
            credit="Rizwan Saleem",
            summary=(
                "1,087,806 comments filtered from the complete August 2026 Reddit dump, "
                "every stage gated and independently recomputed. Forecasts are held-out "
                "August days shown against what actually happened."
            ),
        ),
        Dataset(
            key="real_arctic_2026_10",
            label="Real · Arctic Shift rebuild · Oct–Nov 2026",
            kind="real",
            path=OUTPUTS / "real_arctic_2026_10",
            daily=OUTPUTS / "real_arctic_2026_10" / "daily_series.csv",
            source="539,063 Arctic Shift comments → one contiguous 24-day window, bots and boilerplate filtered",
            window="2026-10-23 → 2026-11-15",
            communities=38,
            forecast="future",
            credit="Aaqib Mehmood",
            summary=(
                "257,129 comments in one contiguous window with the real SNAP hyperlink "
                "graph. The only corpus with forecasts of days after the data, backed by "
                "a held-out backtest against naive baselines."
            ),
        ),
    ]
}

DEFAULT = "real_torrent_2026_08"


def get(key: str) -> Dataset:
    return DATASETS[key]


def real() -> list[Dataset]:
    return [d for d in DATASETS.values() if d.is_real]


def available() -> list[Dataset]:
    return [d for d in DATASETS.values() if d.available()]
