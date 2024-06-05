import logging

import mrc
import mrc.core.operators as ops
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

import cudf

from nv_ingest.schemas.image_filter_schema import ImageFilterSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "filter_images"
MODULE_NAMESPACE = "nv-ingest"
ImageFilterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageFilterSchema)


def _apply_filter(ctrl_msg: ControlMessage, task_params: dict):
    min_size = task_params.get("min_size")
    max_aspect_ratio = task_params.get("max_aspect_ratio")
    min_aspect_ratio = task_params.get("min_aspect_ratio")
    filter_flag = task_params.get("filter", False)

    with ctrl_msg.payload().mutable_dataframe() as mdf:
        # return if no images
        image_mask = mdf["document_type"] == ContentTypeEnum.IMAGE.value
        if not image_mask.any():
            return ctrl_msg

        # detect undesirable images
        base_cols = mdf.columns
        gdf_image = mdf.loc[image_mask]

        img_width = gdf_image["metadata"].struct.field("image_metadata").struct.field("width")

        img_height = gdf_image["metadata"].struct.field("image_metadata").struct.field("height")

        avg_size = (img_width + img_height) / 2
        aspect_ratio = (img_width / img_height).fillna(0)

        image_filter_mask = ~(
            (avg_size > min_size) & (aspect_ratio < max_aspect_ratio) & (aspect_ratio > min_aspect_ratio)
        )

        if image_filter_mask.any():
            # if we want do immediately remove undesireable images from payload
            if filter_flag:
                # Slow first time, jitify is performs a one-time only warm-up to populate the persistent cache.
                result_gdf = mdf[base_cols].drop(labels=gdf_image.loc[image_filter_mask].index, inplace=False)
                # Strange segfault if we don't do this...
                result_gdf = cudf.from_pandas(result_gdf.to_pandas())
                message_meta = MessageMeta(df=result_gdf)
                ctrl_msg.payload(message_meta)

                return ctrl_msg

            # explode to extract individual metadata structs
            mdf_temp = mdf["metadata"].struct.explode()
            exploded_metadata_cols = list(mdf_temp.columns)
            mdf[exploded_metadata_cols] = mdf_temp
            filtered_images_gdf = gdf_image.loc[image_filter_mask]

            # define and validate `info_message_metadata`
            info_msg = {
                "task": TaskTypeEnum.FILTER.value,
                "status": StatusEnum.SUCCESS.value,
                "message": "Filtered due to image size.",
                "filter": True,
            }

            validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).dict()

            # update payload with `info_message_metadata` and `document_type`
            filtered_images_gdf["info_message_metadata"] = [validated_info_msg] * filtered_images_gdf.shape[0]
            mdf.drop(labels=["info_message_metadata", "metadata"], inplace=True, axis=1)
            mdf["info_message_metadata"] = filtered_images_gdf["info_message_metadata"]
            mdf.loc[filtered_images_gdf["document_type"].index, "document_type"] = ContentTypeEnum.INFO_MSG.value
            mdf["metadata"] = mdf[exploded_metadata_cols + ["info_message_metadata"]].to_struct()
            mdf.drop(labels=mdf.columns.difference(base_cols), inplace=True, axis=1)

    return ctrl_msg


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _filter_images(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, ImageFilterSchema)

    @filter_by_task(["filter"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def filter_images_fn(ctrl_msg: ControlMessage):
        # based on this reference:
        # https://gitlab-master.nvidia.com/daustin/govdocs_ingest/-/blob/main/govdocs_ingest.py?ref_type=heads#L258

        task_props = ctrl_msg.remove_task("filter")
        filter_type = task_props.get("type")
        task_params = task_props.get("params")

        if filter_type != ContentTypeEnum.IMAGE:
            return ctrl_msg

        ctrl_msg = _apply_filter(ctrl_msg, task_params)

        return ctrl_msg

    # Create a node for filtering incoming images
    input_node = builder.make_node(
        "image_filter",
        ops.map(filter_images_fn),
    )

    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)
