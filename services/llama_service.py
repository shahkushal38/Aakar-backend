# llama_service.py - Service for interacting with Llama Cloud
import os
from llama_cloud.client import LlamaCloud
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import NodeWithScore
from services.chat_service import upload_files, get_files_project
from typing import List

client = LlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))

def create_llama_index(project_id):
    print(f"Creating Llama Cloud index for project: {project_id} os.getenv: {os.getenv('LLAMA_CLOUD_API_KEY')}")
    embedding_config = {
        'type': 'OPENAI_EMBEDDING',
        'component': {
            'api_key': os.getenv("OPENAI_API_KEY"),
            'model_name': 'text-embedding-3-large'
        }
    }
    transform_config = {
        'mode': 'advanced',
        'segmentation_config': {
            'mode': 'page',
        },
        'chunking_config': {
            'mode': 'token',
            'chunk_size': '1024',
            'chunk_overlap': '200'
        }
    }
    llama_parse_parameters = {
        'premium_mode': True,
        'extract_charts': True,
        'use_vendor_multimodal_model': False,
        'auto_mode': True,
        'auto_mode_trigger_on_table_in_page': True,
        'auto_mode_trigger_on_image_in_page': True,
        'annotate_links': True,
        'guess_xlsx_sheet_name': True,
    }
    preset_retrieval_parameters = {
        'retrieval_mode': 'files_via_content',
        'files_top_k': 10
    }
    pipeline = {
        'name': f"project_{project_id}_index",
        'embedding_config': embedding_config,
        'transform_config': transform_config,
        'llama_parse_parameters': llama_parse_parameters
    }
    pipeline = client.pipelines.upsert_pipeline(request=pipeline)
    return pipeline

def upload_to_llama_cloud(index_id, file_path, project_id):
    with open(file_path, 'rb') as f:
        file = client.files.upload_file(upload_file=f)
    files = [{'file_id': file.id}]
    print(f"Uploading file to Llama Cloud index: {index_id}, file: {file_path}")
    pipeline_files = client.pipelines.add_files_to_pipeline(index_id, request=files)
    # upload to firebase
    upload_files(project_id, file.id)
    return pipeline_files

def upload_to_llama_cloud_only(file_path):
    with open(file_path, 'rb') as f:
        file = client.files.upload_file(upload_file=f)
    return file

def list_files_in_llama_cloud(project_id):
    print(f"Listing files in Llama Cloud for project: {project_id}")
    files = client.files.list_files()
    serialized_files = [
        {
            "id": file.id,
            "name": file.name,
            "file_size": file.file_size,
            "file_type": file.file_type,
            "project_id": file.project_id,
            "created_at": file.created_at.isoformat() if hasattr(file.created_at, "isoformat") else str(file.created_at),
            "updated_at": file.updated_at.isoformat() if hasattr(file.updated_at, "isoformat") else str(file.updated_at),
            "last_modified_at": file.last_modified_at.isoformat() if hasattr(file.last_modified_at, "isoformat") else str(file.last_modified_at),
            "resource_info": file.resource_info,
            "permission_info": file.permission_info,
            "data_source_id": file.data_source_id,
        }
        for file in files
    ]

    print(f"Serialized files: {serialized_files}")

    # filter by project_id
    firbasefiles = get_files_project(project_id)
    # firbasefiles schema
    # file = {
    #     "project_id": project_id,
    #     "file_id": file.id,
    #     "timestamp": get_timestamp()
    # }
    # filter serialized_files by all ids of firbasefiles
    serialized_files = [file for file in serialized_files if file["id"] in [f["file_id"] for f in firbasefiles]]
    return serialized_files

def initialize_agent(project_id: str):
    index = LlamaCloudIndex(
        name=f"project_{project_id}_index",
        project_name="Default",
        api_key=os.getenv("LLAMA_CLOUD_API_KEY")
    )
    doc_retriever = index.as_retriever(
        retrieval_mode="files_via_content",
        files_top_k=5
    )
    chunk_retriever = index.as_retriever(
        retrieval_mode="chunks",
        rerank_top_n=5
    )
    llm = OpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2048
    )

    def chunk_retriever_fn(query: str) -> List[NodeWithScore]:
        return chunk_retriever.retrieve(query)

    def doc_retriever_fn(query: str) -> List[NodeWithScore]:
        return doc_retriever.retrieve(query)

    chunk_retriever_tool = FunctionTool.from_defaults(fn=chunk_retriever_fn, name="chunk_retriever_fn")
    doc_retriever_tool = FunctionTool.from_defaults(fn=doc_retriever_fn, name="doc_retriever_fn")

    return {
        "chunk_tool": chunk_retriever_tool,
        "doc_tool": doc_retriever_tool,
        "llm": llm
    }
