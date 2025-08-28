from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi

hub_api = HfApi()
repo_id  = "SrikrishnaIyer/so100_picknplace_100episodes"   
root = Path("/home/krishna/.cache/huggingface/lerobot/SrikrishnaIyer/so100_picknplace_100episodes")
hub_api.create_repo(repo_id=repo_id, private=False, repo_type="dataset",exist_ok=True)
upload_kwargs = {
            "repo_id": repo_id,
            "folder_path": root,
            "repo_type": "dataset"
        }

hub_api.upload_large_folder(**upload_kwargs)