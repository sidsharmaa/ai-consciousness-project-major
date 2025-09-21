"""
Configuration loader for the project.

Uses Pydantic to load and validate settings from a YAML file,
ensuring that the configuration is type-safe and structured correctly.
"""
from pathlib import Path
from typing import List
from typing import Dict

import yaml
from pydantic import BaseModel

# Define the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DataSourceConfig(BaseModel):
    """Configuration for the data source."""
    kaggle_dataset: str

class ProcessingConfig(BaseModel):
    """Configuration for data processing parameters."""
    keywords: List[str]

class PathConfig(BaseModel):
    """Configuration for project paths."""
    download_dir: Path
    output_path: Path

class LocalJsonProcessingConfig(BaseModel):
    """Configuration for processing the local arXiv JSON snapshot."""
    input_path: Path
    output_path: Path
    filter_keywords: List[str]
    target_categories: List[str]
    max_title_len: int
    max_abstract_len: int

class TextSplitterConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class EmbeddingPipelineConfig(BaseModel):
    transcript_sources: List[Path]
    parquet_source: Path
    faiss_index_path: Path
    embedding_model: str
    text_splitter: TextSplitterConfig

class LLMConfig(BaseModel):
    model_name: str
    base_url: str

class RAGApplicationConfig(BaseModel):
    faiss_index_path: Path
    log_path: Path
    embedding_model: str
    llm: LLMConfig
    prompt_template: str
    answer_length_map: Dict[str, int]

class AppConfig(BaseModel):
    """Main application configuration model."""
    data_source: DataSourceConfig
    processing: ProcessingConfig
    paths: PathConfig
    local_json_processing: LocalJsonProcessingConfig
    embedding_pipeline: EmbeddingPipelineConfig
    rag_application: RAGApplicationConfig

def load_config() -> AppConfig:
    """
    Loads configuration from the YAML file and returns a validated AppConfig object.

    Returns:
        AppConfig: The validated application configuration.
    """
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)

    # Resolve relative paths to absolute paths for all path-containing sections
    path_sections = ["paths", "local_json_processing"]
    for section_name in path_sections:
        if section_name in config_yaml:
            for key, value in config_yaml[section_name].items():
                if "path" in key or "dir" in key:
                    if isinstance(value, str):
                        config_yaml[section_name][key] = PROJECT_ROOT / value
    
    return AppConfig.model_validate(config_yaml)

# Load the configuration globally on import
config = load_config()