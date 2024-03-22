import pytest

from nv_ingest.util.tracing.tagging import traceable


class MockControlMessage:
    def __init__(self):
        self.metadata = {}

    def has_metadata(self, key):
        return key in self.metadata

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def set_metadata(self, key, value):
        self.metadata[key] = value


@pytest.fixture
def mock_control_message():
    return MockControlMessage()


# Test with trace tagging enabled and custom trace name
def test_traceable_with_trace_tagging_enabled_custom_name(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", True)

    @traceable(trace_name="CustomTrace")
    def sample_function(message):
        pass  # Function body is not relevant for the test

    sample_function(mock_control_message)

    assert "trace::entry::CustomTrace" in mock_control_message.metadata
    assert "trace::exit::CustomTrace" in mock_control_message.metadata
    assert isinstance(mock_control_message.metadata["trace::entry::CustomTrace"], int)
    assert isinstance(mock_control_message.metadata["trace::exit::CustomTrace"], int)


# Test with trace tagging enabled and no custom trace name
def test_traceable_with_trace_tagging_enabled_no_custom_name(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", True)

    @traceable()
    def another_function(message):
        pass  # Function body is not relevant for the test

    another_function(mock_control_message)

    assert "trace::entry::another_function" in mock_control_message.metadata
    assert "trace::exit::another_function" in mock_control_message.metadata
    assert isinstance(
        mock_control_message.metadata["trace::entry::another_function"], int
    )
    assert isinstance(
        mock_control_message.metadata["trace::exit::another_function"], int
    )


# Test with trace tagging disabled
def test_traceable_with_trace_tagging_disabled(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", False)

    @traceable()
    def disabled_function(message):
        pass  # Function body is not relevant for the test

    disabled_function(mock_control_message)

    # Ensure no trace metadata was added since trace tagging was disabled
    assert not any(key.startswith("trace::") for key in mock_control_message.metadata)
