# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import inspect
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.graph.pipeline_graph import Graph, Node


def _ensure_event_loop() -> None:
    """Guarantee an asyncio event loop exists for the current thread.

    uvloop >= 0.22's ``get_event_loop()`` raises ``RuntimeError`` when no loop
    is *running*, which breaks Ray Data's async-actor initialisation in freshly
    spawned worker processes.  We fall back to the default asyncio policy (which
    allows implicit loop creation) and explicitly create + set a loop so that
    Ray's ``get_or_create_event_loop()`` finds one immediately.
    """
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        asyncio.set_event_loop(asyncio.new_event_loop())


# Run at import time so the loop exists before Ray's _init_async() runs.
_ensure_event_loop()


class AbstractOperator(ABC):
    """Base class for all pipeline operators."""

    def __init__(self, **kwargs: Any) -> None:
        self._graph_init_kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def preprocess(self, data: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def process(self, data: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def postprocess(self, data: Any, **kwargs: Any) -> Any: ...

    def run(self, data: Any, **kwargs: Any) -> Any:
        data = self.preprocess(data, **kwargs)
        data = self.process(data, **kwargs)
        data = self.postprocess(data, **kwargs)
        return data

    async def aprocess(self, data: Any, **kwargs: Any) -> Any:
        """Async version of :meth:`process`.

        The default calls ``process()`` synchronously on the event-loop
        thread.  This is intentional: most operators use C extensions
        (pypdfium2, OpenCV, torch, …) that are **not** thread-safe, so
        ``asyncio.to_thread`` would allow Ray Data to run multiple
        batches in parallel threads inside the same actor, causing
        memory corruption.

        I/O-bound subclasses that call remote endpoints should override
        this with a proper ``await``-based implementation.
        """
        return self.process(data, **kwargs)

    async def arun(self, data: Any, **kwargs: Any) -> Any:
        """Async version of :meth:`run`."""
        data = self.preprocess(data, **kwargs)
        data = await self.aprocess(data, **kwargs)
        data = self.postprocess(data, **kwargs)
        return data

    async def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Make operators directly usable as Ray ``map_batches`` async callables."""
        return await self.arun(data, **kwargs)

    def get_constructor_kwargs(self) -> dict[str, Any]:
        """Best-effort constructor kwargs for executor-side reconstruction."""
        kwargs = dict(getattr(self, "_graph_init_kwargs", {}))
        signature = inspect.signature(type(self).__init__)
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in kwargs:
                continue
            if hasattr(self, name):
                kwargs[name] = getattr(self, name)
                continue
            private_name = f"_{name}"
            if hasattr(self, private_name):
                kwargs[name] = getattr(self, private_name)
        return kwargs

    def __rshift__(self, other: "AbstractOperator | Node") -> "Graph":
        """``operator_a >> operator_b`` — auto-wrap both in Nodes and chain them.

        Returns a :class:`Graph` so the pipeline is immediately usable::

            graph = op_a >> op_b >> op_c
        """
        from nemo_retriever.graph.pipeline_graph import Node

        left = Node(self)
        # Delegate to Node.__rshift__ which returns a Graph
        return left >> other
