from dataclasses import dataclass, field
import os

@dataclass
class DVCConfig:
    """DVC + MinIO 远端配置。优先级：环境变量 > 显式传参 > 默认值。"""

    # GitHub 数据仓库
    github_repo: str = field(
        default_factory=lambda: os.environ.get("DVC_GITHUB_REPO", "")
    )
    github_token: str = field(
        default_factory=lambda: os.environ.get("DVC_GITHUB_TOKEN", "")
    )
    github_api_base: str = field(
        default_factory=lambda: os.environ.get("DVC_GITHUB_API_BASE", "https://api.github.com")
    )

    # MinIO 对象存储
    # TODO: 硬编码后端，未来升级需重新设计
    minio_endpoint: str = field(
        default_factory=lambda: os.environ.get("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.environ.get("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.environ.get("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.environ.get("MINIO_BUCKET", "dvc-store")
    )
    minio_virtual_folder: str = field(
        default_factory=lambda: os.environ.get("MINIO_VIRTUAL_FOLDER", "")
    )
    minio_secure: bool = False

    # 数据拉取行为
    dvc_pattern: str = "dvc_data"   # .dvc 文件前缀过滤
    tag_strategy: str = "release"   # "release" | "latest"
