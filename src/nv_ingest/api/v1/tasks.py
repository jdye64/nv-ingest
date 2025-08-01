from datetime import datetime
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import uuid
from nv_ingest_api.internal.extract.image.image_extractor import extract_primitives_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageConfigSchema
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor import _inject_validated_config
from nv_ingest.framework.orchestration.ray.util.pipeline.stage_builders import get_nim_service
from nv_ingest_api.internal.enums.common import DocumentTypeEnum
from nv_ingest_api.util.converters.type_mappings import doc_type_to_content_type
from nv_ingest_api.internal.enums.common import AccessLevelEnum, ContentTypeEnum, LanguageEnum, TextTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import ContentHierarchySchema
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job
from celery import Celery
import base64
from opentelemetry import trace

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

celery_app = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1"
)


def stage_1_extract_pdf_components(job_spec_dict):
    print("-------- Entering stage_1_extract_pdf_components ---------")

    control_message = IngestControlMessage()
    job_id = None

    try:
        # Validate incoming job structure
        validate_ingest_job(job_spec_dict)

        ts_entry = datetime.now()
        job_id = job_spec_dict.pop("job_id")

        job_payload = job_spec_dict.get("job_payload", {})
        job_tasks = job_spec_dict.get("tasks", [])
        tracing_options = job_spec_dict.pop("tracing_options", {})

        # Extract tracing options
        do_trace_tagging = tracing_options.get("trace", True)
        if do_trace_tagging in (True, "True", "true", "1"):
            do_trace_tagging = True

        ts_send = tracing_options.get("ts_send")
        if ts_send is not None:
            ts_send = datetime.fromtimestamp(ts_send / 1e9)
        trace_id = tracing_options.get("trace_id")

        # Create response channel and load payload
        df = pd.DataFrame(job_payload)
        control_message.payload(df)
        annotate_cm(control_message, message="Created")

        # Add basic metadata
        control_message.set_metadata("job_id", job_id)
        control_message.set_metadata("timestamp", datetime.now().timestamp())

        # Add task definitions to the control message
        for task in job_tasks:
            task_id = task.get("id", str(uuid.uuid4()))
            task_type = task.get("type", "unknown")
            task_props = task.get("task_properties", {})

            if not isinstance(task_props, dict):
                task_props = task_props.model_dump()

            task_obj = ControlMessageTask(
                id=task_id,
                type=task_type,
                properties=task_props,
            )
            print(f"Adding task: {task_obj}")
            control_message.add_task(task_obj)

        # Apply tracing metadata and timestamps if enabled
        control_message.set_metadata("config::add_trace_tagging", do_trace_tagging)
        if do_trace_tagging:
            ts_exit = datetime.now()

            control_message.set_timestamp("trace::entry::message_broker_task_source", ts_entry)
            control_message.set_timestamp("trace::exit::message_broker_task_source", ts_exit)

            if ts_send is not None:
                control_message.set_timestamp("trace::entry::broker_source_network_in", ts_send)

            if trace_id is not None:
                if isinstance(trace_id, int):
                    trace_id = trace.format_trace_id(trace_id)
                control_message.set_metadata("trace_id", trace_id)

            control_message.set_timestamp("latency::ts_send", datetime.now())

        print(f"Message processed successfully with job_id: {job_id}")

    except Exception as e:
        print(f"Failed to process job submission: {e}")

        if job_id is not None:
            response_channel = f"{job_id}"
            control_message.set_metadata("job_id", job_id)
            control_message.set_metadata("response_channel", response_channel)
            control_message.set_metadata("cm_failed", True)

            annotate_cm(control_message, message="Failed to process job submission", error=str(e))
        else:
            raise

    return control_message
    

def stage_2_metadata_injection(control_message: IngestControlMessage):
    df = control_message.payload()
    update_required = False
    rows = []
    print(f"Starting metadata injection on DataFrame with {len(df)} rows")

    for _, row in df.iterrows():
        try:
            # Convert document type to content type using enums.
            content_type = doc_type_to_content_type(DocumentTypeEnum(row["document_type"]))
            # Check if metadata is missing or doesn't contain 'content'
            if (
                "metadata" not in row
                or not isinstance(row["metadata"], dict)
                or "content" not in row["metadata"].keys()
            ):
                update_required = True

                # Initialize default structures based on MetaDataSchema
                default_source_metadata = {
                    "source_id": row.get("source_id"),
                    "source_name": row.get("source_name"),
                    "source_type": row["document_type"],
                    "source_location": "",
                    "collection_id": "",
                    "date_created": datetime.now().isoformat(),
                    "last_modified": datetime.now().isoformat(),
                    "summary": "",
                    "partition_id": -1,
                    "access_level": AccessLevelEnum.UNKNOWN.value,
                }

                default_content_metadata = {
                    "type": content_type.name.lower(),
                    "page_number": -1,
                    "description": "",
                    "hierarchy": ContentHierarchySchema().model_dump(),
                    "subtype": "",
                    "start_time": -1,
                    "end_time": -1,
                }

                default_audio_metadata = None
                if content_type == ContentTypeEnum.AUDIO:
                    default_audio_metadata = {
                        "audio_type": row["document_type"],
                        "audio_transcript": "",
                    }

                default_image_metadata = None
                if content_type == ContentTypeEnum.IMAGE:
                    default_image_metadata = {
                        "image_type": row["document_type"],
                        "structured_image_type": ContentTypeEnum.NONE.value,
                        "caption": "",
                        "text": "",
                        "image_location": (0, 0, 0, 0),
                        "image_location_max_dimensions": (0, 0),
                        "uploaded_image_url": "",
                        "width": 0,
                        "height": 0,
                    }

                default_text_metadata = None
                if content_type == ContentTypeEnum.TEXT:
                    default_text_metadata = {
                        "text_type": TextTypeEnum.DOCUMENT.value,
                        "summary": "",
                        "keywords": "",
                        "language": LanguageEnum.UNKNOWN.value,
                        "text_location": (0, 0, 0, 0),
                        "text_location_max_dimensions": (0, 0, 0, 0),
                    }

                row["metadata"] = {
                    "content": row["content"],
                    "content_metadata": default_content_metadata,
                    "error_metadata": None,
                    "audio_metadata": default_audio_metadata,
                    "image_metadata": default_image_metadata,
                    "source_metadata": default_source_metadata,
                    "text_metadata": default_text_metadata,
                }
                print(
                    f"METADATA_INJECTOR_DEBUG: Rebuilt metadata for source_id='{row.get('source_id', 'N/A')}'. "
                    f"Metadata keys: {list(row['metadata'].keys())}."
                    f"'content' present: {'content' in row['metadata']}"
                )
        except Exception as inner_e:
            print(f"Failed to process row during metadata injection: {inner_e}")
            raise inner_e
        rows.append(row)

    if update_required:
        docs = pd.DataFrame(rows)
        control_message.payload(docs)
        print(f"Metadata injection updated payload with {len(docs)} rows")
    else:
        print("No metadata update was necessary during metadata injection")

    return control_message


def stage_3_pdf_extraction(control_message):
    print("-------- Entering stage_3_pdf_extraction ---------")
    
    # The validated config. I don't like setting this here.
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    nemoretriever_parse_grpc, nemoretriever_parse_http, nemoretriever_parse_auth, nemoretriever_parse_protocol = (
        get_nim_service("nemoretriever_parse")
    )
    model_name = os.environ.get("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")
    
    print(f"Yolox grpc: {yolox_grpc}")
    print(f"Yolox http: {yolox_http}")
    print(f"Yolox auth: {yolox_auth}")
    print(f"Yolox protocol: {yolox_protocol}")
    print(f"Nemoretriever parse grpc: {nemoretriever_parse_grpc}")
    print(f"Nemoretriever parse http: {nemoretriever_parse_http}")
    print(f"Nemoretriever parse auth: {nemoretriever_parse_auth}")

    validated_config = PDFExtractorSchema(
        **{
            "pdfium_config": {
                "auth_token": yolox_auth,  # All auth tokens are the same for the moment
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
            },
            "nemoretriever_parse_config": {
                "auth_token": nemoretriever_parse_auth,
                "nemoretriever_parse_endpoints": (nemoretriever_parse_grpc, nemoretriever_parse_http),
                "nemoretriever_parse_infer_protocol": nemoretriever_parse_protocol,
                "nemoretriever_parse_model_name": model_name,
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
            },
        }
    )
    
    # Extract the DataFrame payload.
    df_extraction_ledger = control_message.payload()
    print(f"Extracted payload with {len(df_extraction_ledger)} rows.")

    # Remove the "extract" task from the message to obtain task-specific configuration.
    task_config = remove_task_by_type(control_message, "extract")
    print(f"Extracted task config: {task_config}")

    # Perform PDF extraction.
    execution_trace_log = {}
    new_df, extraction_info = _inject_validated_config(
        df_extraction_ledger,
        task_config,
        execution_trace_log=execution_trace_log,
        validated_config=validated_config,
    )
    print(f"PDF extraction completed. Extracted {len(new_df)} rows.")

    # Update the message payload with the extracted DataFrame.
    control_message.payload(new_df)
    # Optionally, annotate the message with extraction info.
    control_message.set_metadata("pdf_extraction_info", extraction_info)
    print("PDF extraction metadata injected successfully.")

    do_trace_tagging = control_message.get_metadata("config::add_trace_tagging") is True
    if do_trace_tagging and execution_trace_log:
        for key, ts in execution_trace_log.items():
            control_message.set_timestamp(key, ts)

    return control_message


def stage_4_image_extraction(control_message):
    print("-------- Entering stage_4_image_extraction ---------")
    
    # Hate this ...
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")

    validated_config = ImageConfigSchema(
        **{
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
            "auth_token": yolox_auth,  # All auth tokens are the same for the moment
        }
    )

    print("ImageExtractorStage.on_data: Starting image extraction process.")
    try:
        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        print(f"Extracted payload with {len(df_ledger)} rows.")

        # Remove the "extract" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract")
        print(f"Extracted task config: {task_config}")

        # Perform image primitives extraction.
        new_df, extraction_info = extract_primitives_from_image_internal(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            extraction_config=validated_config,
            execution_trace_log=None,
        )
        print(f"Image extraction completed. Resulting DataFrame has {len(new_df)} rows.")

        # Update the message payload with the extracted primitives DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("image_extraction_info", extraction_info)

        return control_message
    except Exception as e:
        print(f"ImageExtractorStage failed processing control message: {e}")
        raise


# Hate, move
def _extract_data_frame(message: Any) -> Tuple[Any, Any]:
    """
    Extracts a DataFrame from a message payload and returns it along with selected columns.
    """
    try:
        df = message.payload()
        print(f"Sink received DataFrame with {len(df)} rows.")
        keep_cols = ["document_type", "metadata"]
        return df, df[keep_cols].to_dict(orient="records")
    except Exception as err:
        print(f"Failed to extract DataFrame: {err}")
        return None, None


def _split_large_dict(json_data: List[Dict[str, Any]], size_limit: int) -> List[List[Dict[str, Any]]]:
    fragments = []
    current_fragment = []
    current_size = sys.getsizeof(json.dumps(current_fragment))
    for item in json_data:
        item_size = sys.getsizeof(json.dumps(item))
        if current_size + item_size > size_limit:
            fragments.append(current_fragment)
            current_fragment = []
            current_size = sys.getsizeof(json.dumps(current_fragment))
        current_fragment.append(item)
        current_size += item_size
    if current_fragment:
        fragments.append(current_fragment)
    return fragments


# Hate, move
def _create_json_payload(message: Any, df_json: Any) -> List[Dict[str, Any]]:
    # df_json_str = json.dumps(df_json)
    # df_json_size = sys.getsizeof(df_json_str)
    # size_limit = 128 * 1024 * 1024  # 128 MB limit
    # if df_json_size > size_limit:
    #     data_fragments = _split_large_dict(df_json, size_limit)
    #     fragment_count = len(data_fragments)
    # else:
    #     data_fragments = [df_json]
    #     fragment_count = 1
    
    data_fragments = [df_json]
    fragment_count = 1

    ret_val_json_list = []
    for i, fragment_data in enumerate(data_fragments):
        ret_val_json = {
            "status": "success" if not message.get_metadata("cm_failed", False) else "failed",
            "description": (
                "Successfully processed the message."
                if not message.get_metadata("cm_failed", False)
                else "Failed to process the message."
            ),
            "data": fragment_data,
            "fragment": i,
            "fragment_count": fragment_count,
        }
        if i == 0 and message.get_metadata("add_trace_tagging", True):
            trace_snapshot = message.filter_timestamp("trace::")
            ret_val_json["trace"] = {key: ts.timestamp() * 1e9 for key, ts in trace_snapshot.items()}
            ret_val_json["annotations"] = {
                key: message.get_metadata(key) for key in message.list_metadata() if key.startswith("annotation::")
            }
        ret_val_json_list.append(ret_val_json)
    print(f"Sink created {len(ret_val_json_list)} JSON payloads.")
    return ret_val_json_list


def stage_final_prepare_response(control_message):
    print("-------- Entering stage_final_prepare_response ---------")
    mdf, df_json = None, None
    json_result_fragments = []
    # response_channel = control_message.get_metadata("response_channel")
    try:
        cm_failed = control_message.get_metadata("cm_failed", False)
        if not cm_failed:
            mdf, df_json = _extract_data_frame(control_message)
            json_result_fragments = _create_json_payload(control_message, df_json)
        else:
            json_result_fragments = _create_json_payload(control_message, None)

        total_payload_size = 0
        json_payloads = []
        for i, fragment in enumerate(json_result_fragments, start=1):
            payload = json.dumps(fragment)
            size_bytes = len(payload.encode("utf-8"))
            total_payload_size += size_bytes
            size_mb = size_bytes / (1024 * 1024)
            print(f"Sink Fragment {i} size: {size_mb:.2f} MB")
            json_payloads.append(payload)

        total_size_mb = total_payload_size / (1024 * 1024)
        print(f"Sink Total JSON payload size: {total_size_mb:.2f} MB")
        annotate_cm(control_message, message="Pushed")
        # self._push_to_broker(json_payloads, response_channel)

    except ValueError as e:
        mdf_size = len(mdf) if mdf is not None and not mdf.empty else 0
        # self._handle_failure(response_channel, json_result_fragments, e, mdf_size)
        print(f"Critical error processing message: {e}")
    except Exception as e:
        # logger.exception(f"Critical error processing message: {e}")
        mdf_size = len(mdf) if mdf is not None and not mdf.empty else 0
        # self._handle_failure(response_channel, json_result_fragments, e, mdf_size)
        print(f"Critical error processing message: {e}")

    # return control_message
    return json_payloads



def stage_final_prepare_response_simplified(control_message):
    print("-------- Entering stage_final_prepare_response ---------")
    mdf, df_json = None, None

    cm_failed = control_message.get_metadata("cm_failed", False)
    if not cm_failed:
        mdf, df_json = _extract_data_frame(control_message)
        json_result_fragments = _create_json_payload(control_message, df_json)
    else:
        json_result_fragments = _create_json_payload(control_message, None)
        
    return json_result_fragments[0]


@celery_app.task
def process_pdf(job_spec_dict):
    print("-------- Entering process_pdf ---------")

    # Stage 1: Extract PDF components - message_broker_task_source.py:_process_message() adaptation for celery
    control_message = stage_1_extract_pdf_components(job_spec_dict)
    
    # Stage 2: Metadata Injection - metadata_injector.py:on_data()
    control_message = stage_2_metadata_injection(control_message)
    
    # Stage 3: PDF Extraction - pdf_extractor.py:on_data()
    control_message = stage_3_pdf_extraction(control_message)
    
    # # Stage 4: Image Extraction - image_extractor.py:on_data()
    # control_message = stage_4_image_extraction(control_message)
    
    # Stage Final: Prepare response and send to Redis - message_broker_task_sink.py:on_data()
    response = stage_final_prepare_response_simplified(control_message)
    
    return response

