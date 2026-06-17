from dataclasses import dataclass, field
import os

@dataclass
class DVCConfig:
    """DVC + SeaweedFS 远端配置。优先级：环境变量 > 显式传参 > 默认值。"""

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

    # SeaweedFS(S3) 对象存储
    seaweedfs_endpoint: str = field(
        default_factory=lambda: os.environ.get("SEAWEEDFS_ENDPOINT", "")
    )
    seaweedfs_access_key: str = field(
        default_factory=lambda: os.environ.get("SEAWEEDFS_ACCESS_KEY", "")
    )
    seaweedfs_secret_key: str = field(
        default_factory=lambda: os.environ.get("SEAWEEDFS_SECRET_KEY", "")
    )
    seaweedfs_bucket: str = field(
        default_factory=lambda: os.environ.get("SEAWEEDFS_BUCKET", "dvc-store")
    )
    seaweedfs_virtual_folder: str = field(
        default_factory=lambda: os.environ.get("SEAWEEDFS_VIRTUAL_FOLDER", "")
    )
    seaweedfs_secure: bool = False

    # 数据拉取行为
    dvc_pattern: str = "dvc_data"   # .dvc 文件前缀过滤
    tag_strategy: str = "release"   # "release" | "latest"
