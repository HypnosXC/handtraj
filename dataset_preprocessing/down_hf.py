

from huggingface_hub import login, snapshot_download

# 1. Login with your token (found in HF Settings -> Access Tokens)
login(token=token)

# 2. Download the entire repository (model or dataset)
# Replace 'repo_id' with the target (e.g., 'meta-llama/Llama-2-7b')
local_path = snapshot_download(
    repo_id="Franklin0/private_handdata", 
    repo_type="dataset",
    local_dir="/data-share/share/handdata/preprocessed/"
)

print(f"Files downloaded to: {local_path}")