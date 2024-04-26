import pytest
from nv_ingest_client.primitives.tasks.store import StoreTask

# Initialization and Property Setting


def test_store_task_initialization():
    task = StoreTask(
        content_type="text",
        store_method="s3",
        endpoint="minio:9000",
        access_key="foo",
        secret_key="bar",
    )
    assert task._content_type == "text"
    assert task._store_method == "s3"
    assert task._extra_params["endpoint"] == "minio:9000"
    assert task._extra_params["access_key"] == "foo"
    assert task._extra_params["secret_key"] == "bar"


# String Representation Tests


def test_store_task_str_representation():
    task = StoreTask(content_type="image", store_method="minio", endpoint="localhost:9000")
    expected_str = "Store Task:\n" "  content type: image\n" "  store method: minio\n" "  endpoint: localhost:9000\n"
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "content_type, store_method, extra_param_1, extra_param_2",
    [
        ("image", "minio", "foo", "bar"),
        ("text", "s3", "foo", False),
        ("table", None, 1, True),
        (None, None, 2, "foo"),  # Test default parameters
    ],
)
def test_store_task_to_dict(
    content_type,
    store_method,
    extra_param_1,
    extra_param_2,
):
    task = StoreTask(
        content_type=content_type,
        store_method=store_method,
        extra_param_1=extra_param_1,
        extra_param_2=extra_param_2,
    )

    expected_dict = {"type": "store", "task_properties": {"params": {}}}

    expected_dict["task_properties"]["content_type"] = content_type or "image"
    expected_dict["task_properties"]["method"] = store_method or "minio"
    expected_dict["task_properties"]["params"]["extra_param_1"] = extra_param_1
    expected_dict["task_properties"]["params"]["extra_param_2"] = extra_param_2

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"
