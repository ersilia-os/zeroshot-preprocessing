import sys
from typing import Optional

from rich.console import Console
from loguru import logger as _loguru

# Note: do NOT call _loguru.remove() here — that would remove all sinks
# registered by other packages sharing the same global loguru instance.

# Loguru format: timestamp | level | message
_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "{message}"
)


class Logger:
    def __init__(self):
        self._loguru = _loguru
        # Dedicated Rich console on stderr for structured output.
        self._console = Console(stderr=True, highlight=False)
        self._sink_id: Optional[int] = None
        self._verbose = False

    def set_verbosity(self, verbose: bool):
        self._verbose = verbose
        if verbose and self._sink_id is None:
            self._sink_id = self._loguru.add(
                sys.stderr,
                format=_FORMAT,
                colorize=True,
                level="DEBUG",
            )
        elif not verbose and self._sink_id is not None:
            try:
                self._loguru.remove(self._sink_id)
            except Exception:
                pass
            self._sink_id = None

    # ------------------------------------------------------------------
    # Plain-text log levels
    # ------------------------------------------------------------------

    def debug(self, msg: str):
        self._loguru.debug(msg)

    def info(self, msg: str):
        self._loguru.info(msg)

    def warning(self, msg: str):
        self._loguru.warning(msg)

    def error(self, msg: str):
        self._loguru.error(msg)

    def success(self, msg: str):
        self._loguru.success(msg)

    # ------------------------------------------------------------------
    # Structured Rich output (only emitted when verbose=True)
    # ------------------------------------------------------------------

    def rule(self, title: str = "", style: str = "dim blue"):
        if not self._verbose:
            return
        if title:
            self._console.rule(f"[bold cyan]{title}[/]", style=style)
        else:
            self._console.rule(style=style)

    def profile_summary(self, profile) -> None:
        if not self._verbose:
            return

        task_label = {
            "classification": "Classification",
            "regression":     "Regression",
        }.get(profile.task, profile.task)

        parts = [
            f"n={profile.n_samples:,}",
            f"p={profile.n_features:,}",
            f"n/p={profile.n_p_ratio:.1f}",
        ]
        if profile.task == "regression":
            parts.append(f"y_skewness={profile.y_skewness:.2f}")
        if profile.is_sparse_counts:
            parts.append("sparse_counts=True")
        if profile.binary_feature_fraction > 0.5:
            parts.append(f"binary_frac={profile.binary_feature_fraction:.2f}")
        parts.append(f"skewness={profile.median_feature_skewness:.2f}")
        parts.append(f"outliers={profile.outlier_fraction:.2f}")
        if profile.near_zero_variance_fraction > 0.01:
            parts.append(f"nzv={profile.near_zero_variance_fraction:.2f}")
        parts.append(f"corr={profile.median_abs_correlation:.2f}")

        sep = "  [dim]|[/]  "
        body = sep.join(f"[cyan]{p}[/]" for p in parts)
        self._console.print(f"[bold]{task_label}[/bold]  {body}")


logger = Logger()
