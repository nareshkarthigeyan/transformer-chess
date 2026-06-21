import os
from huggingface_hub import HfApi

api = HfApi()

# Let's see if Python can actually see the file first
filename = "checkpoint.pt"  # Change this if it's named 'model.pt' or inside a folder like 'outputs/checkpoint.pt'

if not os.path.exists(filename):
    print(f"Error: Could not find '{filename}' in your current folder!")
    print("Files available here are:", os.listdir("."))
else:
    print(f"Found {filename}! Starting direct upload to Hugging Face...")
    
    url = api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id="nareshkarthigeyan/intuition1",
        commit_message="Forcing model weight checkpoint upload"
    )
    
    print(f" Upload absolutely complete! View it here: {url}")