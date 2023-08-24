import os
import warnings
from torch import nn
import tempfile
import torch


class BaseModel(nn.Module):
    def to_folder(self, path: os.PathLike):
        if os.path.exists(path):
            warnings.warn(f"Folder {path} already exists. Overwriting.")

        else:
            os.mkdir(path)

        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        self.config.to_json(path)

    @classmethod
    def from_folder(cls, path: os.PathLike):
        config = cls.Config.from_json(path)
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        return model

    def to_huggingface(
        self,
        repo_id: str = None,
        token: str = None,
        private: bool = False,
        commit_message: str = None,
        exist_ok: bool = False,
        run_as_future: bool = False,
    ):
        try:
            import huggingface_hub as hf_hub
        except ImportError as e:
            raise ImportError(
                "Please install huggingface_hub to use this function."
            ) from e

        hf_hub.create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=exist_ok,
            repo_type="model",
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.to_folder(tmpdirname)
            api = hf_hub.HfApi()
            api.upload_folder(
                folder_path=tmpdirname,
                repo_id=repo_id,
                repo_type="model",
                token=token,
                commit_message=commit_message,
            )

    @classmethod
    def from_huggingface(cls, repo_id: str, token: str = None):
        try:
            import huggingface_hub as hf_hub
        except ImportError as e:
            raise ImportError(
                "Please install huggingface_hub to use this function."
            ) from e

        folder_path = hf_hub.snapshot_download(repo_id, token=token)
        return cls.from_folder(folder_path)
