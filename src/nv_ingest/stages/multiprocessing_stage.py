import ctypes
import logging
import multiprocessing as mp
import os
import queue
import threading as mt
import time
import typing
import uuid

import mrc
import pandas as pd
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from mrc import SegmentObject
from mrc.core import operators as ops
from mrc.core.subscriber import Observer

import cudf

from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task

logger = logging.getLogger(f"morpheus.{__name__}")


class MultiProcessingBaseStage(SinglePortStage):
    """
    A ControlMessage oriented base multiprocessing stage to increase parallelism of stages written in Python.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        Integer for how many process engines to use for pdf content extraction.
    max_queue_size : int
        Integer for how large to make pe_engine queues.
    process_fn : typing.Callable[[pd.DataFrame, dict], pd.DataFrame]
        The function that will be executed in each process enginer. The function will
        accept a pandas dataframe from a ControlMessage payload and dictionary of task arguements.

    Returns
    -------
    cudf.DataFrame
        A cuDF dataframe.
    """

    # TODO implement dataframe filter_fn support for splitting like stages.

    def __init__(
        self,
        c: Config,
        task: str,
        task_desc: str,
        pe_count: int,
        process_fn: typing.Callable[[pd.DataFrame, dict], pd.DataFrame],
    ):
        super().__init__(c)
        self._task = task
        self._task_desc = task_desc
        self._pe_count = pe_count
        self._process_fn = process_fn
        self._max_queue_size = 1
        self._mp_context = mp.get_context("fork")
        self._cancellation_token = self._mp_context.Value(ctypes.c_int8, False)
        self._pass_thru_recv_queue = queue.Queue(maxsize=c.edge_buffer_size)
        self._my_threads = {}
        self._ctrl_msg_ledger = {}

    @property
    def name(self) -> str:
        return self._task

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage,)

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    @staticmethod
    def child_receive(
        recv_queue: mp.Queue,
        send_queue: mp.Queue,
        cancellation_token: mp.Value,
        process_fn: typing.Callable[[pd.DataFrame, dict], pd.DataFrame],
    ):
        while not cancellation_token.value:
            # get work from recv_queue
            try:
                event = recv_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event["type"] == "on_next":
                work_package = event["value"]
                df = work_package["payload"]
                task_props = work_package["task_props"]
                result = process_fn(df, task_props)
                work_package["payload"] = result
                send_queue.put({"type": "on_next", "value": work_package})
                continue

            if event["type"] == "on_error":
                send_queue.put(event)
                break

            if event["type"] == "on_completed":
                send_queue.put(event)
                break

        # this on_completed may be unnecessary/unexpected if we've already forwarded an on_completed event.
        send_queue.put({"type": "on_completed"})

    @staticmethod
    def parent_receive(
        mp_context,
        max_queue_size,
        send_queue: mp.Queue,
        sub: mrc.Subscriber,
        cancellation_token: mp.Value,
        process_fn: typing.Callable[[pd.DataFrame, dict], pd.DataFrame],
    ):
        recv_queue = mp_context.Queue(maxsize=max_queue_size)

        process_engine = mp_context.Process(
            target=MultiProcessingBaseStage.child_receive, args=(send_queue, recv_queue, cancellation_token, process_fn)
        )

        process_engine.start()

        while not cancellation_token.value and sub.is_subscribed():
            # get completed work
            try:
                event = recv_queue.get_nowait()  # (timeout=0.1)
            except queue.Empty:
                continue

            if event["type"] == "on_next":
                sub.on_next(event["value"])
                continue

            if event["type"] == "on_error":
                sub.on_next(event["value"])
                break

            if event["type"] == "on_completed":
                sub.on_completed()
                break

        sub.on_completed()

    def obserable_fn(self, obs: mrc.Observable, sub: mrc.Subscriber):
        send_queue = self._mp_context.Queue(maxsize=self._max_queue_size)

        tid = str(uuid.uuid4())
        self._my_threads[tid] = mt.Thread(
            target=MultiProcessingBaseStage.parent_receive,
            args=(self._mp_context, self._max_queue_size, send_queue, sub, self._cancellation_token, self._process_fn),
        )

        @nv_ingest_node_failure_context_manager(
            annotation_id=self.name,
            raise_on_failure=False,
        )
        def forward_fn(ctrl_msg: ControlMessage):
            # TODO extend traceable decorator to include entry/exit options
            ts_fetched = time.time_ns()

            do_trace_tagging = (ctrl_msg.has_metadata("config::add_trace_tagging") is True) and (
                ctrl_msg.get_metadata("config::add_trace_tagging") is True
            )

            if do_trace_tagging:
                ts_send = ctrl_msg.get_metadata("latency::ts_send", None)
                ts_entry = time.time_ns()
                ctrl_msg.set_metadata(f"trace::entry::{self._task_desc}", ts_entry)
                if ts_send:
                    ctrl_msg.set_metadata(f"trace::entry::{self._task_desc}_channel_in", ts_send)
                    ctrl_msg.set_metadata(f"trace::exit::{self._task_desc}_channel_in", ts_fetched)

            while True:
                try:
                    self._pass_thru_recv_queue.put(ctrl_msg, timeout=0.1)
                    break
                except queue.Full:
                    continue

            return ctrl_msg

        @filter_by_task([("extract", {"document_type": "pdf"})], forward_func=forward_fn)
        @nv_ingest_node_failure_context_manager(
            annotation_id=self.name,
            raise_on_failure=False,
        )
        def on_next(ctrl_msg: ControlMessage):
            # TODO extend traceable decorator to include entry/exit options
            ts_fetched = time.time_ns()

            do_trace_tagging = (ctrl_msg.has_metadata("config::add_trace_tagging") is True) and (
                ctrl_msg.get_metadata("config::add_trace_tagging") is True
            )

            if do_trace_tagging:
                ts_send = ctrl_msg.get_metadata("latency::ts_send", None)
                ts_entry = time.time_ns()
                ctrl_msg.set_metadata(f"trace::entry::{self._task_desc}", ts_entry)
                if ts_send:
                    ctrl_msg.set_metadata(f"trace::entry::{self._task_desc}_channel_in", ts_send)
                    ctrl_msg.set_metadata(f"trace::exit::{self._task_desc}_channel_in", ts_fetched)

            with ctrl_msg.payload().mutable_dataframe() as mdf:
                df = mdf.to_pandas()

            task_props = ctrl_msg.get_tasks().get("extract").pop()
            cm_id = uuid.uuid4()
            self._ctrl_msg_ledger[cm_id] = ctrl_msg
            work_package = {}
            work_package["payload"] = df
            work_package["task_props"] = task_props
            work_package["cm_id"] = cm_id
            send_queue.put({"type": "on_next", "value": work_package})

        def on_error(error: BaseException):
            logger.debug(f"obs on error {os.getpid()}")
            send_queue.put({"type": "on_error", "value": error})

        def on_completed():
            logger.debug(f"obs on completed {os.getpid()}")
            send_queue.put({"type": "on_completed"})

        self._my_threads[tid].start()

        obs.subscribe(Observer.make_observer(on_next, on_error, on_completed))

        self._my_threads[tid].join()

    def _build_single(self, builder: mrc.Builder, input_node: SegmentObject) -> SegmentObject:
        def reconstruct_fn(work_package):
            ctrl_msg = self._ctrl_msg_ledger.pop(work_package["cm_id"])

            @nv_ingest_node_failure_context_manager(
                annotation_id=self.name,
                raise_on_failure=False,
            )
            def cm_func(ctrl_msg: ControlMessage, work_package: dict):
                gdf = cudf.from_pandas(work_package["payload"])
                ctrl_msg.payload(MessageMeta(df=gdf))

                return ctrl_msg

            return cm_func(ctrl_msg, work_package)

        def pass_thru_source_fn():
            while True:
                try:
                    ctrl_msg = self._pass_thru_recv_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                yield ctrl_msg

        @nv_ingest_node_failure_context_manager(
            annotation_id=self.name,
            raise_on_failure=False,
        )
        def merge_fn(ctrl_msg: ControlMessage):
            # TODO extend traceable decorator to include entry/exit options
            do_trace_tagging = (ctrl_msg.has_metadata("config::add_trace_tagging") is True) and (
                ctrl_msg.get_metadata("config::add_trace_tagging") is True
            )

            if do_trace_tagging:
                ts_exit = time.time_ns()
                ctrl_msg.set_metadata(f"trace::exit::{self._task_desc}", ts_exit)
                ctrl_msg.set_metadata("latency::ts_send", ts_exit)

            return ctrl_msg

        # worker branch
        worker_node = builder.make_node(f"{self.name}-worker-fn", mrc.core.operators.build(self.obserable_fn))
        worker_node.launch_options.pe_count = self._pe_count
        reconstruct_node = builder.make_node(f"{self.name}-reconstruct", ops.map(reconstruct_fn))

        # create merge node
        merge_node = builder.make_node(
            "merge",
            ops.map(merge_fn),
        )

        # pass thru source
        pass_thru_source = builder.make_source(f"{self.name}-pass-thru-source", pass_thru_source_fn)

        builder.make_edge(input_node, worker_node)
        builder.make_edge(worker_node, reconstruct_node)
        builder.make_edge(reconstruct_node, merge_node)
        builder.make_edge(pass_thru_source, merge_node)

        return merge_node

    async def join(self):
        logger.debug("stopping...")
        self._cancellation_token.value = True
        for _, thread in self._my_threads.items():
            thread.join()
        await super().join()
        logger.debug("stopped")
