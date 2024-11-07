import datasets
from typing import Optional

def load_dataset_by_type(dataset_type: str = "small", split: str = "train") -> datasets.Dataset:
    """Load either the small benchmark or full GAIA dataset"""
    if dataset_type.lower() == "small":
        ds = datasets.load_dataset("m-ric/agents_small_benchmark")[split]
    elif dataset_type.lower() == "gaia":
        ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[split]
    else:
        raise ValueError("dataset_type must be 'small' or 'gaia'")
    
    if "answer" in ds.column_names:
        ds = ds.rename_column("answer", "true_answer")
    
    def preprocess_file_paths(row):
        if row.get("file_name", "") and len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{split}/" + row["file_name"]
        return row
    
    return ds.map(preprocess_file_paths) 