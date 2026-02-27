from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

import pandas as pd

T = TypeVar("T")


def repeated_row_values(
    row_count: int,
    *,
    value: Any = None,
    factory: Callable[[], Any] | None = None,
) -> list[Any]:
    if factory is not None:
        return [factory() for _ in range(int(row_count))]
    return [value for _ in range(int(row_count))]


def apply_dataframe_defaults(
    df: pd.DataFrame,
    *,
    column_defaults: Mapping[str, Any | Callable[[], Any]],
) -> pd.DataFrame:
    out = df.copy()
    n_rows = len(out.index)
    for col, default in column_defaults.items():
        if callable(default):
            out[col] = repeated_row_values(n_rows, factory=default)
        else:
            out[col] = repeated_row_values(n_rows, value=default)
    return out


def execute_actor_call(
    batch: Any,
    *,
    invoke: Callable[[], T],
    dataframe_error: Callable[[pd.DataFrame, BaseException], T],
    other_error: Callable[[BaseException], T],
) -> T:
    try:
        return invoke()
    except BaseException as exc:
        if isinstance(batch, pd.DataFrame):
            return dataframe_error(batch, exc)
        return other_error(exc)
