# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``@configured`` decorator — registry metadata and runtime config injection."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel

from nemo_retriever.config.context import try_get_config
from nemo_retriever.config.justification import ConfigJustification, normalize_justifications
from nemo_retriever.config.registry import build_configured_entry, register_configured

F = TypeVar("F", bound=Callable[..., Any])

_REDACTED_FIELD_NAMES = frozenset({"api_key", "api_token", "password", "secret"})


def get_config_section(config: BaseModel, dotted_path: str) -> Any:
    """Resolve a dotted path such as ``pipeline_defaults.extract`` on *config*."""
    current: Any = config
    for part in dotted_path.split("."):
        if current is None:
            return None
        if isinstance(current, BaseModel):
            current = getattr(current, part, None)
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            raise TypeError(f"Cannot traverse {dotted_path!r} through {type(current)!r}")
    return current


def _resolve_param_name(fn: Callable[..., Any], model: type[BaseModel], param_name: str | None) -> str | None:
    if param_name is not None:
        return param_name
    sig = inspect.signature(fn)
    for name, param in sig.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            continue
        ann = param.annotation
        if ann is model:
            return name
        origin = getattr(ann, "__origin__", None)
        if origin is not None and model in getattr(ann, "__args__", ()):
            return name
    return None


def _inject_config_value(
    bound_args: inspect.BoundArguments,
    *,
    param_name: str | None,
    model: type[BaseModel],
    section: str,
) -> None:
    if param_name is None:
        return
    current = bound_args.arguments.get(param_name, inspect.Parameter.empty)
    if current not in (inspect.Parameter.empty, None):
        return
    config = try_get_config()
    if config is None:
        return
    raw = get_config_section(config, section)
    if raw is None:
        return
    if isinstance(raw, model):
        bound_args.arguments[param_name] = raw
    elif isinstance(raw, BaseModel):
        bound_args.arguments[param_name] = model.model_validate(raw.model_dump())
    elif isinstance(raw, dict):
        bound_args.arguments[param_name] = model.model_validate(raw)
    else:
        bound_args.arguments[param_name] = model.model_validate(raw)


def configured(
    *,
    section: str,
    model: type[BaseModel],
    param_name: str | None = None,
    variants: Sequence[str] = (),
    justification: ConfigJustification | Sequence[ConfigJustification] = (),
    rationale: str = "",
    run_modes: Sequence[str] = ("inprocess", "batch", "service"),
) -> Callable[[F], F]:
    """Declare and optionally inject configuration for *fn*.

    When *justification* is non-empty, *rationale* must explain the tradeoff.
    """
    tags = normalize_justifications(justification)
    if tags and not rationale.strip():
        raise ValueError("@configured requires a non-empty rationale when justification tags are set")

    def decorator(fn: F) -> F:
        resolved_param = _resolve_param_name(fn, model, param_name)
        register_configured(
            build_configured_entry(
                fn=fn,
                section=section,
                model=model,
                param_name=resolved_param,
                variants=tuple(variants),
                justification=tags,
                rationale=rationale,
                run_modes=tuple(run_modes),
            )
        )

        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            _inject_config_value(bound, param_name=resolved_param, model=model, section=section)
            return fn(*bound.args, **bound.kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
