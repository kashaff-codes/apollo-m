"""
Temporal Fusion Transformer forecaster for APOLLO-M macro-layer forecasting.

Uses pytorch-forecasting's native TFT with QuantileLoss for prediction intervals.
Text embeddings from BERT/RoBERTa are injected as continuous time-varying covariates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.decomposition import PCA

log = logging.getLogger("APOLLO-M")

FORECAST_QUANTILES = [0.1, 0.5, 0.9]

# A subreddit needs at least this share of the days in its calendar span to be modelled.
MIN_OBSERVED_RATIO = 0.6


@dataclass
class TFTConfig:
    lookback_window: int = 14
    forecast_horizon: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 32
    attention_head_size: int = 2
    dropout: float = 0.2
    hidden_continuous_size: int = 16
    max_epochs: int = 15
    embed_pca_components: int = 32
    model_dir: Path = field(default_factory=lambda: Path("models"))
    quantiles: list[float] = field(default_factory=lambda: FORECAST_QUANTILES.copy())
    # Daily comment counts can be a scraper artifact (page-size multiples). The real-data
    # pipeline turns this off; the default keeps the original behaviour (benchmark scripts).
    use_comment_count: bool = True


class TFTForecaster:
    """PyTorch Lightning training pipeline for Temporal Fusion Transformer."""

    def __init__(self, config: Optional[TFTConfig] = None):
        self.config = config or TFTConfig()
        self.model: Optional[TemporalFusionTransformer] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None
        self.pca: Optional[PCA] = None
        self.embedding_cols: list[str] = []
        self.raw_embedding_cols: list[str] = []
        self.origin_date: Optional[pd.Timestamp] = None   # calendar day that time_idx == 0
        self.cutoff_idx: Optional[int] = None             # last time_idx seen in training
        self.last_backtest: Optional[pd.DataFrame] = None

    def prepare_dataframe(self, daily_df: pd.DataFrame, origin: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Normalize columns for TimeSeriesDataSet consumption.

        time_idx is the CALENDAR day number (days since `origin`), not a row counter, so a
        missing day stays a gap instead of being squeezed away.
        """
        df = daily_df.copy()
        df["subreddit"] = df["subreddit"].astype(str).apply(
            lambda x: x if x.startswith("r/") else f"r/{x}"
        )
        df["group_id"] = df["subreddit"]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values(["subreddit", "date"]).reset_index(drop=True)
        day = df["date"].dt.normalize()
        if origin is None:
            origin = day.min()
        df["time_idx"] = (day - origin).dt.days.astype(int)

        for col, default in [("avg_toxicity", 0.5), ("toxic_rate", 0.4)]:
            if col not in df.columns:
                df[col] = default
            df[col] = df[col].fillna(default).clip(0, 1)

        if "comment_count" not in df.columns:
            df["comment_count"] = 1
        df["comment_count"] = df["comment_count"].fillna(1).clip(lower=1)

        return df

    def _resolve_embedding_columns(self, df: pd.DataFrame) -> list[str]:
        return sorted(
            [col for col in df.columns if col.startswith("embed_")],
            key=lambda name: int(name.split("_", 1)[1]),
        )

    def fit_embedding_pca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce high-dimensional text embeddings for TFT covariate injection."""
        self.raw_embedding_cols = self._resolve_embedding_columns(df)
        if not self.raw_embedding_cols:
            self.embedding_cols = []
            return df

        matrix = df[self.raw_embedding_cols].fillna(0.0).values
        n_components = min(
            self.config.embed_pca_components,
            matrix.shape[0],
            matrix.shape[1],
        )
        if n_components < 1:
            self.embedding_cols = []
            return df

        self.pca = PCA(n_components=n_components, random_state=42)
        reduced = self.pca.fit_transform(matrix)
        self.embedding_cols = [f"emb_pca_{i}" for i in range(n_components)]
        for idx, col in enumerate(self.embedding_cols):
            df[col] = reduced[:, idx]

        explained = self.pca.explained_variance_ratio_.sum()
        log.info(
            "PCA reduced %d-dim embeddings → %d TFT covariates (%.1f%% variance)",
            len(self.raw_embedding_cols),
            n_components,
            explained * 100,
        )
        return df

    def transform_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted PCA to new data."""
        if self.pca is None or not self.raw_embedding_cols:
            return df

        available = [col for col in self.raw_embedding_cols if col in df.columns]
        if not available:
            return df

        matrix = df[available].fillna(0.0).values
        reduced = self.pca.transform(matrix)
        for idx, col in enumerate(self.embedding_cols):
            df[col] = reduced[:, idx]
        return df

    def _trained_subreddit_labels(self) -> set:
        """Subreddit labels known to the fitted TimeSeriesDataSet encoders."""
        encoders = getattr(self.training_dataset, "_categorical_encoders", {})
        for key in ("__group_id__subreddit", "subreddit"):
            encoder = encoders.get(key)
            if encoder is None or not hasattr(encoder, "classes_"):
                continue
            labels = set(encoder.classes_.keys())
            labels.discard("nan")
            labels.discard(np.nan)
            return labels
        return set()

    def _snapshot_training_state(self) -> dict:
        """Preserve TFT artifacts so holdout evaluation cannot clobber production state."""
        return {
            "model": self.model,
            "training_dataset": self.training_dataset,
            "pca": self.pca,
            "embedding_cols": list(self.embedding_cols),
            "raw_embedding_cols": list(self.raw_embedding_cols),
        }

    def _restore_training_state(self, snapshot: dict) -> None:
        self.model = snapshot["model"]
        self.training_dataset = snapshot["training_dataset"]
        self.pca = snapshot["pca"]
        self.embedding_cols = snapshot["embedding_cols"]
        self.raw_embedding_cols = snapshot["raw_embedding_cols"]

    def _eligible_counts(self, df: pd.DataFrame) -> pd.Series:
        """Observed-day counts for subreddits with enough calendar span and few enough gaps."""
        need = self.config.lookback_window + self.config.forecast_horizon
        g = df.groupby("subreddit")["time_idx"].agg(["min", "max", "count"])
        span = g["max"] - g["min"] + 1
        ok = (span >= need) & (g["count"] >= MIN_OBSERVED_RATIO * span)
        return g.loc[ok, "count"].sort_values(ascending=False)

    def eligible_subreddits(self, daily_df: pd.DataFrame) -> list[str]:
        """'r/<name>' labels of every subreddit with enough history to forecast."""
        return self._eligible_counts(self.prepare_dataframe(daily_df)).index.tolist()

    def _forecast_subreddit_candidates(
        self, df: pd.DataFrame, trained_subs: set, preferred: str | None = None
    ) -> list[str]:
        eligible = self._eligible_counts(df)
        if trained_subs:
            eligible = eligible[eligible.index.isin(trained_subs)]

        candidates = eligible.index.astype(str).tolist()
        if preferred and preferred in candidates:
            return [preferred] + [s for s in candidates if s != preferred]
        return candidates

    def build_dataset(self, df: pd.DataFrame, for_training: bool = True) -> TimeSeriesDataSet:
        """Create a TimeSeriesDataSet with subreddit as the entity identifier."""
        max_pred = self.config.forecast_horizon
        max_encoder = self.config.lookback_window

        min_required = max_encoder + max_pred
        valid_groups = self._eligible_counts(df).index.tolist()

        if not valid_groups:
            log.warning(
                "No subreddit has a >= %d-day calendar span with enough observed days — skipping TFT.",
                min_required,
            )
            return pd.DataFrame()

        working = df[df["subreddit"].isin(valid_groups)].copy()
        if for_training:
            cutoff = working["time_idx"].max() - max_pred
            working = working[working["time_idx"] <= cutoff]
            self.cutoff_idx = int(cutoff)   # the last max_pred days are held out of training

        time_varying_unknown = ["avg_toxicity", "toxic_rate"]
        if self.config.use_comment_count:
            time_varying_unknown.append("comment_count")
        time_varying_unknown.extend(self.embedding_cols)

        # With missing days the encoder window (lookback calendar days) holds fewer than
        # lookback real observations. Allow a shorter encoder then. Gap-free data (the
        # benchmark) keeps the original full-length encoder.
        counts = working.groupby("subreddit")["time_idx"].agg(["min", "max", "count"])
        has_gaps = bool((counts["count"] < counts["max"] - counts["min"] + 1).any())
        min_encoder = max(1, max_encoder // 2) if has_gaps else max_encoder

        self.training_dataset = TimeSeriesDataSet(
            working,
            time_idx="time_idx",
            target="avg_toxicity",
            group_ids=["subreddit"],
            min_encoder_length=min_encoder,
            max_encoder_length=max_encoder,
            max_prediction_length=max_pred,
            time_varying_unknown_reals=time_varying_unknown,
            time_varying_known_reals=["time_idx"],
            static_categoricals=["subreddit"],
            categorical_encoders={"subreddit": NaNLabelEncoder(add_nan=True)},
            allow_missing_timesteps=True,
        )
        return self.training_dataset

    def train(self, daily_df: pd.DataFrame) -> None:
        """Train TFT via PyTorch Lightning."""
        df = self.prepare_dataframe(daily_df)
        self.origin_date = df["date"].dt.normalize().min()
        df = self.fit_embedding_pca(df)
        dataset = self.build_dataset(df, for_training=True)
        if not isinstance(dataset, TimeSeriesDataSet):
            # build_dataset returns an empty DataFrame when no subreddit has enough history.
            log.warning("TFT training skipped: not enough daily history.")
            self.model = None
            return

        train_loader = dataset.to_dataloader(
            train=True,
            batch_size=self.config.batch_size,
            num_workers=0,
        )

        quantile_loss = QuantileLoss(quantiles=self.config.quantiles)
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            loss=quantile_loss,
            log_interval=10,
        )

        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="auto",
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(self.model, train_dataloaders=train_loader)

        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.config.model_dir / "tft_model.pt")
        log.info("TFT trained and saved → %s", self.config.model_dir / "tft_model.pt")

    def predict_quantiles(
        self, daily_df: pd.DataFrame, subreddit: str | None = None, future: bool = False
    ) -> dict:
        """
        Generate multi-quantile forecasts.

        Returns dict with keys: quantiles, forecast (horizon x n_quantiles),
        p10, p50, p90 arrays of length forecast_horizon, and subreddit.

        future=False (default, original behaviour): predicts the LAST forecast_horizon days that
        are already in daily_df (a hindcast). future=True: predicts the forecast_horizon days
        AFTER the last observed day, which is what a real forecast means.
        """
        if self.model is None or self.training_dataset is None:
            raise RuntimeError("TFT model not trained. Call train() first.")

        df = self.prepare_dataframe(daily_df, origin=self.origin_date)
        df = self.transform_embeddings(df)

        trained_subs = self._trained_subreddit_labels()
        if trained_subs:
            df = df[df["subreddit"].isin(trained_subs)].copy()
        if df.empty:
            return {}

        candidates = self._forecast_subreddit_candidates(df, trained_subs, subreddit)
        for forecast_sub in candidates:
            sub_df = df[df["subreddit"] == forecast_sub].sort_values("time_idx")
            if future:
                sub_df = self._with_future_rows(sub_df, int(sub_df["time_idx"].max()))
            forecast = self._predict_array(sub_df, forecast_sub)
            if forecast is None:
                continue

            q_idx = {q: i for i, q in enumerate(self.config.quantiles)}
            log.info("TFT forecast generated for %s", forecast_sub)
            return {
                "quantiles": self.config.quantiles,
                "forecast": forecast,
                "p10": forecast[:, q_idx.get(0.1, 0)],
                "p50": forecast[:, q_idx.get(0.5, 1)],
                "p90": forecast[:, q_idx.get(0.9, 2)],
                "subreddit": forecast_sub,
            }

        return {}

    def _with_future_rows(self, sub_df: pd.DataFrame, anchor_idx: int) -> pd.DataFrame:
        """History up to anchor_idx plus forecast_horizon placeholder rows for the days after it.

        The placeholder rows only carry the dates / time_idx to predict; the TFT decoder does not
        read their unknown values (they are copies of the last observed row so nothing is NaN).
        """
        horizon = self.config.forecast_horizon
        hist = sub_df[sub_df["time_idx"] <= anchor_idx]
        fut = pd.concat([hist.iloc[[-1]]] * horizon, ignore_index=True)
        fut["time_idx"] = anchor_idx + np.arange(1, horizon + 1)
        fut["date"] = self.origin_date + pd.to_timedelta(fut["time_idx"], unit="D")
        return pd.concat([hist, fut], ignore_index=True)

    def _predict_array(self, sub_df: pd.DataFrame, label: str) -> Optional[np.ndarray]:
        """Quantile forecast [horizon, n_quantiles] for the last max_prediction_length steps of sub_df."""
        try:
            pred_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                sub_df,
                predict=True,
                stop_randomization=True,
            )
        except AssertionError as exc:
            log.warning("TFT prediction window unavailable for %s: %s", label, exc)
            return None

        pred_loader = pred_dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size,
            num_workers=0,
        )

        # mode="quantiles" returns the full p10/p50/p90 spread (QuantileLoss).
        raw_preds = self.model.predict(pred_loader, mode="quantiles", return_x=False)
        preds = raw_preds.detach().cpu().numpy()

        # Expected shape: [n_samples, horizon, n_quantiles]. One group per call, so take the
        # last sample; tolerate a squeezed 2-D [horizon, n_quantiles] batch of one.
        if preds.ndim == 3:
            forecast = preds[-1]
        elif preds.ndim == 2:
            forecast = preds
        else:  # 1-D fallback: single-step, tile across quantiles
            forecast = preds.reshape(-1, 1).repeat(len(self.config.quantiles), axis=1)

        forecast = forecast[: self.config.forecast_horizon, :]
        forecast = np.clip(forecast, 0.0, 1.0)
        # Enforce p10 <= p50 <= p90 per day (quantile crossing guard).
        return np.sort(forecast, axis=1)

    def backtest(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Out-of-sample check on the days the model never trained on.

        The model is trained only up to cutoff_idx, so the last forecast_horizon calendar days
        are held out. For every subreddit we forecast them from the earlier history and compare
        with what actually happened, next to two naive baselines (repeat the last value, repeat
        the recent average). One row per subreddit.
        """
        if self.model is None or self.training_dataset is None or self.cutoff_idx is None:
            return pd.DataFrame()

        horizon = self.config.forecast_horizon
        anchor = self.cutoff_idx
        q = {qq: i for i, qq in enumerate(self.config.quantiles)}
        df = self.prepare_dataframe(daily_df, origin=self.origin_date)
        df = self.transform_embeddings(df)
        trained = self._trained_subreddit_labels()

        rows = []
        for sub in sorted(df["subreddit"].unique()):
            if trained and sub not in trained:
                continue
            s = df[df["subreddit"] == sub].sort_values("time_idx")
            hist = s[s["time_idx"] <= anchor]
            actual = s[(s["time_idx"] > anchor) & (s["time_idx"] <= anchor + horizon)]
            if hist.empty or actual.empty:
                continue
            span = anchor - hist["time_idx"].min() + 1
            if span < self.config.lookback_window or len(hist) < MIN_OBSERVED_RATIO * span:
                continue
            fc = self._predict_array(self._with_future_rows(s, anchor), sub)
            if fc is None:
                continue
            step = actual["time_idx"].to_numpy() - anchor - 1
            y = actual["avg_toxicity"].to_numpy(float)
            p10, p50, p90 = fc[step, q.get(0.1, 0)], fc[step, q.get(0.5, 1)], fc[step, q.get(0.9, 2)]
            recent = hist[hist["time_idx"] > anchor - self.config.lookback_window]["avg_toxicity"]

            def pin(quant: float, pred: np.ndarray) -> float:
                err = y - pred
                return float(np.mean(np.maximum(quant * err, (quant - 1) * err)))

            rows.append({
                "subreddit": sub,
                "n_days": int(len(y)),
                "mae_tft": float(np.mean(np.abs(p50 - y))),
                "mae_last_value": float(np.mean(np.abs(hist["avg_toxicity"].iloc[-1] - y))),
                "mae_recent_mean": float(np.mean(np.abs(recent.mean() - y))),
                "pinball": float(np.mean([pin(0.1, p10), pin(0.5, p50), pin(0.9, p90)])),
                "inside_p10_p90": float(np.mean((y >= p10) & (y <= p90))),
            })
        return pd.DataFrame(rows)

    def evaluate(self, daily_df: pd.DataFrame, holdout_days: Optional[int] = None) -> dict:
        """Out-of-sample metrics from backtest() (holdout_days is kept only for compatibility)."""
        empty = {"mae": None, "pinball_loss": None, "n_holdout": 0}
        if self.model is None or self.training_dataset is None:
            return empty
        bt = self.backtest(daily_df)
        self.last_backtest = bt
        if bt.empty:
            return empty
        w = bt["n_days"]
        return {
            "mae": float(np.average(bt["mae_tft"], weights=w)),
            "pinball_loss": float(np.average(bt["pinball"], weights=w)),
            "n_holdout": int(w.sum()),
            "n_subreddits": int(len(bt)),
            "baseline_last_value_mae": float(np.average(bt["mae_last_value"], weights=w)),
            "baseline_recent_mean_mae": float(np.average(bt["mae_recent_mean"], weights=w)),
        }

    def save_forecast_csv(
        self,
        forecast: dict,
        subreddit: str,
        start_date: pd.Timestamp,
        output_path: Path,
        write: bool = True,
    ) -> pd.DataFrame:
        """Write multi-quantile forecast results to CSV."""
        horizon = len(forecast["p50"])
        dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        rows = []
        for day_idx in range(horizon):
            p50 = forecast["p50"][day_idx]
            rows.append(
                {
                    "day": day_idx + 1,
                    "date": dates[day_idx].strftime("%Y-%m-%d"),
                    "predicted_toxicity": round(float(p50), 4),
                    "p10": round(float(forecast["p10"][day_idx]), 4),
                    "p50": round(float(p50), 4),
                    "p90": round(float(forecast["p90"][day_idx]), 4),
                    "interval_width": round(
                        float(forecast["p90"][day_idx] - forecast["p10"][day_idx]), 4
                    ),
                    "risk_level": (
                        "CRITICAL"
                        if p50 > 0.8
                        else "HIGH"
                        if p50 > 0.6
                        else "MEDIUM"
                        if p50 > 0.4
                        else "LOW"
                    ),
                    "method": "TFT",
                    "subreddit": subreddit,
                }
            )

        result_df = pd.DataFrame(rows)
        if write:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            log.info("Forecast saved (TFT quantiles) → %s", output_path)
        return result_df