import json

import pytest

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if MORPHEUS_IMPORT_OK and CUDA_DRIVER_OK:
    from morpheus.messages import ControlMessage

    from nv_ingest.modules.sources.redis_task_source import process_message

MODULE_NAME = "redis_task_source"


# Sample job payload
@pytest.fixture
def job_payload():
    return json.dumps(
        {
            "job_payload": {
                "content": ["sample content"],
                "source_name": ["source1"],
                "source_id": ["id1"],
                "document_type": ["pdf"],
            },
            "job_id": "12345",
            "tasks": [
                {
                    "type": "split",
                    "task_properties": {
                        "split_by": "word",
                        "split_length": 100,
                        "split_overlap": 0,
                    },
                },
                {
                    "type": "extract",
                    "task_properties": {
                        "document_type": "pdf",
                        "method": "OCR",
                        "params": {},
                    },
                },
                {"type": "embed", "task_properties": {"model": "bert", "params": {}}},
            ],
        }
    )


@pytest.fixture
def ts_fetched():
    return 1625072042123456789


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@pytest.mark.parametrize(
    "add_trace_tagging, ts_send",
    [
        (True, 1625072042123451000),
        (
            False,
            1625072042123451000,
        ),  # Add case without ts_send when tracing is disabled
    ],
)
def test_process_message(job_payload, add_trace_tagging, ts_send, ts_fetched):
    payload = json.loads(job_payload)

    # Update tracing options based on parameters
    payload["tracing_options"] = {"trace": add_trace_tagging, "ts_send": ts_send}
    modified_payload = json.dumps(payload)
    result = process_message(modified_payload, ts_fetched)

    # Basic type check for the returned object
    assert isinstance(result, ControlMessage)

    # Check for correct handling of tracing options
    assert result.get_metadata("response_channel") == f"response_{payload['job_id']}"
    assert result.get_metadata("job_id") == payload["job_id"]
    if add_trace_tagging:
        assert result.get_metadata("config::add_trace_tagging") is True
        assert result.get_metadata(f"trace::entry::{MODULE_NAME}") is not None
        assert result.get_metadata(f"trace::exit::{MODULE_NAME}") is not None
        if ts_send is not None:
            assert result.get_metadata("trace::entry::redis_source_network_in") == ts_send
            assert result.get_metadata("trace::exit::redis_source_network_in") == ts_fetched
        assert result.get_metadata("latency::ts_send") is not None
    else:
        assert result.get_metadata("config::add_trace_tagging") is None
        # Assert that tracing-related metadata are not set if tracing is disabled
        assert result.get_metadata(f"trace::entry::{MODULE_NAME}") is None
        assert result.get_metadata(f"trace::exit::{MODULE_NAME}") is None

    # Check for the presence of tasks in the ControlMessage
    tasks = ["split", "extract"]
    for task in tasks:
        assert result.has_task(task), f"Expected task {task['type']} not found in ControlMessage."
