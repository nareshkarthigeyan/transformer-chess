import os
from huggingface_hub import HfApi

api = HfApi()

filename = os.environ.get("HF_CHECKPOINT", "checkpoint.pt")
repo_id = os.environ.get("HF_REPO_ID", "nareshkarthigeyan/intuition1")

if not os.path.exists(filename):
    print(f"Error: Could not find '{filename}' in your current folder!")
    print("Files available here are:", os.listdir("."))
else:
    print(f"Found {filename}! Starting direct upload to Hugging Face...")
    
    url = api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=repo_id,
        commit_message="Update Transformer Chess checkpoint",
    )
    
    print(f" Upload absolutely complete! View it here: {url}")
