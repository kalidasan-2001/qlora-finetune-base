from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="D:/hf-cache/models/TinyLlama-1.1B-Chat",
    local_dir_use_symlinks=False,
    resume_download=True
)
