
import asyncio
from typing import List

class GridModelStorage:
    def __init__(self):
        self._storage = None

    async def start(self):
        if not self._storage:
            await self._init_storage()

    async def _init_storage(self):
        try:
            from auragrid.sdk.storage import get_storage_provider
            self._storage = await get_storage_provider()
        except ImportError:
            # AuraGrid SDK not available, do nothing
            pass
        except Exception as e:
            # Log the error
            print(f"Error initializing storage provider: {e}")

    async def upload_model(self, local_path: str, model_id: str):
        if not self._storage:
            return

        try:
            with open(local_path, "rb") as f:
                await self._storage.write_async(f"models/{model_id}/{local_path.split('/')[-1]}", f.read())
        except Exception as e:
            print(f"Error uploading model {model_id}: {e}")

    async def download_model(self, model_id: str, local_path: str):
        if not self._storage:
            return

        try:
            data = await self._storage.read_async(f"models/{model_id}/{local_path.split('/')[-1]}")
            with open(local_path, "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"Error downloading model {model_id}: {e}")

    async def list_models(self) -> List[str]:
        if not self._storage:
            return []

        try:
            files = await self._storage.list_async("models")
            # Assuming list_async returns a list of file paths
            return [f.split('/')[1] for f in files]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
