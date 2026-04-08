"""Tracing system — @observe decorator, Span, Trace, and Tracer."""

from __future__ import annotations

import functools
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Thread-local tracer context
_local = threading.local()


def get_current_tracer() -> Optional["Tracer"]:
    return getattr(_local, "tracer", None)


@dataclass
class Span:
    """Represents a single component execution within a trace."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    span_type: str = "generic"  # "llm", "retriever", "tool", "generic"
    input: Any = None
    output: Any = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["Span"] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: Optional[float] = None

    def finish(self, output: Any = None, error: Optional[str] = None) -> None:
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        if output is not None:
            self.output = output
        if error is not None:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "type": self.span_type,
            "input": str(self.input)[:500] if self.input else None,
            "output": str(self.output)[:500] if self.output else None,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    """A complete end-to-end request trace containing multiple spans."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_span: Optional[Span] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        self.ended_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metadata": self.metadata,
            "root_span": self.root_span.to_dict() if self.root_span else None,
        }


class Tracer:
    """Manages the current trace and span stack during execution."""

    def __init__(self) -> None:
        self._traces: List[Trace] = []
        self._span_stack: List[Span] = []
        self._current_trace: Optional[Trace] = None

    def start_trace(self, metadata: Optional[Dict[str, Any]] = None) -> Trace:
        trace = Trace(metadata=metadata or {})
        self._current_trace = trace
        self._traces.append(trace)
        return trace

    def start_span(self, name: str, span_type: str = "generic", input: Any = None) -> Span:
        span = Span(name=name, span_type=span_type, input=input)
        if self._span_stack:
            self._span_stack[-1].children.append(span)
        elif self._current_trace:
            self._current_trace.root_span = span
        self._span_stack.append(span)
        return span

    def end_span(self, output: Any = None, error: Optional[str] = None) -> Optional[Span]:
        if not self._span_stack:
            return None
        span = self._span_stack.pop()
        span.finish(output=output, error=error)
        return span

    def end_trace(self) -> Optional[Trace]:
        if self._current_trace:
            self._current_trace.finish()
        return self._current_trace

    @property
    def traces(self) -> List[Trace]:
        return self._traces

    def get_last_trace(self) -> Optional[Trace]:
        return self._traces[-1] if self._traces else None

    def print_last_trace(self) -> None:
        trace = self.get_last_trace()
        if trace:
            import json
            print(json.dumps(trace.to_dict(), indent=2))


def observe(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    span_type: str = "generic",
):
    """
    Decorator to instrument a function for component-level tracing.

    Usage:
        @observe
        def my_retriever(query: str) -> list:
            ...

        @observe(name="CustomName", span_type="llm")
        def my_llm_call(prompt: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if tracer is None:
                return fn(*args, **kwargs)

            input_repr = args[0] if args else (list(kwargs.values())[0] if kwargs else None)
            span = tracer.start_span(name=span_name, span_type=span_type, input=input_repr)
            try:
                result = fn(*args, **kwargs)
                tracer.end_span(output=result)
                return result
            except Exception as e:
                tracer.end_span(error=str(e))
                raise

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def set_tracer(tracer: Tracer) -> None:
    """Set the thread-local tracer (call before running traced code)."""
    _local.tracer = tracer


def clear_tracer() -> None:
    _local.tracer = None
