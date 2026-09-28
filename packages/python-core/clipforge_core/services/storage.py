"""
ClipForge AI — Storage Adapter Interface (v2)
Local-first storage system with S3/MinIO cloud adapter capability.
Defaults to local filesystem storage with zero cloud dependencies.
"""
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

from clipforge_core.config import settings

logger = logging.getLogger(__name__)


class BaseStorageAdapter(ABC):
    """Abstract base storage adapter."""

    @abstractmethod
    def save_file(self, local_source_path: Path, target_key: str) -> str:
        """Save a file into storage and return its access URL/path."""
        pass

    @abstractmethod
    def get_file(self, target_key: str, local_dest_path: Path) -> Path:
        """Download or read a file from storage to local destination."""
        pass

    @abstractmethod
    def delete_file(self, target_key: str) -> bool:
        """Delete a file from storage."""
        pass

    @abstractmethod
    def file_exists(self, target_key: str) -> bool:
        """Check if file exists in storage."""
        pass


class LocalStorageAdapter(BaseStorageAdapter):
    """
    Default Localhost Filesystem Storage Adapter.
    Fast, reliable, zero-latency local disk operation.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or settings.MEDIA_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Strip leading slashes to stay within base_dir
        cleaned = key.lstrip("/\\")
        return self.base_dir / cleaned

    def save_file(self, local_source_path: Path, target_key: str) -> str:
        dest = self._resolve(target_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local_source_path.resolve() != dest.resolve():
            shutil.copy2(str(local_source_path), str(dest))
        logger.info(f"[Storage] Saved local asset: {target_key} -> {dest}")
        clean_key = target_key.lstrip("/\\")
        return f"media/{clean_key}"

    def get_file(self, target_key: str, local_dest_path: Path) -> Path:
        src = self._resolve(target_key)
        if not src.exists():
            raise FileNotFoundError(f"Storage key '{target_key}' not found at {src}")
        if src.resolve() != local_dest_path.resolve():
            local_dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(local_dest_path))
        return local_dest_path

    def delete_file(self, target_key: str) -> bool:
        target = self._resolve(target_key)
        if target.exists():
            target.unlink()
            return True
        return False

    def file_exists(self, target_key: str) -> bool:
        return self._resolve(target_key).exists()


class S3StorageAdapter(BaseStorageAdapter):
    """
    S3 / MinIO / Cloudflare R2 Storage Adapter.
    Activated when S3 credentials are provided, with automatic local fallback.
    """

    def __init__(
        self,
        bucket_name: str = settings.MINIO_BUCKET,
        endpoint_url: Optional[str] = settings.MINIO_ENDPOINT,
        access_key: Optional[str] = settings.MINIO_ROOT_USER,
        secret_key: Optional[str] = settings.MINIO_ROOT_PASSWORD,
    ):
        self.bucket_name = bucket_name
        self.local_fallback = LocalStorageAdapter()
        self.is_connected = False
        
        if boto3 and endpoint_url and access_key and secret_key:
            try:
                self.s3 = boto3.client(
                    's3',
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
                # Create bucket if not exists
                try:
                    self.s3.head_bucket(Bucket=bucket_name)
                except ClientError:
                    self.s3.create_bucket(Bucket=bucket_name)
                self.is_connected = True
                logger.info(f"[Storage] Connected to MinIO/S3 bucket: {bucket_name}")
            except Exception as e:
                logger.warning(f"[Storage] Failed to connect to S3: {e}. Falling back to local.")
        else:
            logger.warning("[Storage] Boto3 not installed or S3 credentials missing. Using local fallback.")

    def save_file(self, local_source_path: Path, target_key: str) -> str:
        # Always save to local fallback as well so local processes can find it
        local_url = self.local_fallback.save_file(local_source_path, target_key)
        
        if self.is_connected:
            clean_key = target_key.lstrip("/\\")
            try:
                self.s3.upload_file(str(local_source_path), self.bucket_name, clean_key)
                logger.info(f"[Storage] Uploaded {local_source_path.name} to S3 {self.bucket_name}/{clean_key}")
            except Exception as e:
                logger.error(f"[Storage] Failed to upload to S3: {e}")
                
        return local_url

    def get_file(self, target_key: str, local_dest_path: Path) -> Path:
        clean_key = target_key.lstrip("/\\")
        
        # If it's already in the local cache, just use it
        if self.local_fallback.file_exists(target_key):
            return self.local_fallback.get_file(target_key, local_dest_path)
            
        if self.is_connected:
            try:
                local_dest_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket_name, clean_key, str(local_dest_path))
                logger.info(f"[Storage] Downloaded {clean_key} from S3")
                return local_dest_path
            except Exception as e:
                logger.error(f"[Storage] Failed to download from S3: {e}")
                
        return self.local_fallback.get_file(target_key, local_dest_path)

    def delete_file(self, target_key: str) -> bool:
        local_del = self.local_fallback.delete_file(target_key)
        if self.is_connected:
            clean_key = target_key.lstrip("/\\")
            try:
                self.s3.delete_object(Bucket=self.bucket_name, Key=clean_key)
                return True
            except Exception:
                pass
        return local_del

    def file_exists(self, target_key: str) -> bool:
        if self.local_fallback.file_exists(target_key):
            return True
        if self.is_connected:
            clean_key = target_key.lstrip("/\\")
            try:
                self.s3.head_object(Bucket=self.bucket_name, Key=clean_key)
                return True
            except ClientError:
                return False
        return False


# Global default storage singleton (S3 with Local Fallback)
default_storage = S3StorageAdapter()
