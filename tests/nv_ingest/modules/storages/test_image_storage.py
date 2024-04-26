import json

import pytest
from minio import Minio

from nv_ingest.modules.storages.image_storage import upload_images
from nv_ingest.schemas.metadata_schema import ContentTypeEnum

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if CUDA_DRIVER_OK and MORPHEUS_IMPORT_OK:
    import pandas as pd


class MockMinioClient:
    def __init__(self, *args, **kwargs):
        pass

    def make_bucket(self, *args, **kwargs):
        return

    def put_object(self, *args, **kwargs):
        return

    def bucket_exists(self, *args, **kwargs):
        return True


@pytest.fixture
def mock_minio(mocker):
    def mock_minio_init(
        cls,
        *args,
        **kwargs,
    ):
        return MockMinioClient(*args, **kwargs)

    patched = mocker.patch.object(Minio, "__new__", new=mock_minio_init)
    yield patched


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_upload_images(mock_minio):
    df = pd.DataFrame(
        {
            "document_type": [
                json.dumps(ContentTypeEnum.TEXT.value),
                json.dumps(ContentTypeEnum.IMAGE.value),
            ],
            "metadata": [
                json.dumps({"content": "some text"}),
                json.dumps(
                    {
                        "content": "image_content",
                        "image_metadata": {
                            "image_type": "png",
                        },
                        "source_metadata": {
                            "source_id": "foo",
                        },
                    }
                ),
            ],
        }
    )
    params = {
        "content_type": "image",
    }

    result = upload_images(df, params)
    uploaded_image_url = result.iloc[1]["metadata"]["image_metadata"]["uploaded_image_url"]
    assert uploaded_image_url == "http://localhost:9000/nv-ingest/foo/1.png"
