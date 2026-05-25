from pathlib import Path
from het_ai.studio.bundle import DataBundle
from het_ai.dvc.config import DVCConfig

class DVCLoader:
    """
    将 mlops_demo.GitHubDVCLoader 的能力提炼为框架原生组件。
    
    职责：
      1. 认证 GitHub，解析 release tag
      2. 下载 .dvc 指针文件
      3. 配置 DVC remote（MinIO/S3），执行 dvc pull + checkout
      4. 将版本元数据注入 DataBundle.meta

    用户侧使用方式：
      def load_data(self, dvc_data_root: str) -> DataBundle:
          from het_ai.dvc import DVCLoader, DVCConfig

          loader = DVCLoader(DVCConfig())                    # 从环境变量读取，零配置
          tag, sha = loader.pull(Path(dvc_data_root))

          bundle = DataBundle(...)                           # 用户自己组装数据
          return loader.enrich_bundle(bundle, tag, sha)      # 注入版本元数据，一行搞定
    """

    def __init__(self, config: DVCConfig): ...

    def pull(self, output_path: Path) -> tuple[str, str]:
        """
        执行完整的数据拉取流程。
        
        Returns:
            (tag, commit_sha) — 此次拉取对应的数据版本标识
        """
        # 内部逻辑直接复用 mlops_demo.GitHubDVCLoader 的实现
        # 关键：返回 (tag, commit_sha) 而非 None
        ...

    def enrich_bundle(self, bundle: DataBundle, tag: str, commit_sha: str) -> DataBundle:
        """
        将 DVC 版本元数据注入 DataBundle.meta。
        调用方在 load_data() 末尾调用一次即可。
        """
        bundle.meta.update({
            "dvc_version":    tag,
            "dvc_commit_sha": commit_sha,
            "dvc_repo":       self.config.github_repo,
            "dvc_remote":     f"s3://{self.config.minio_bucket}/{self.config.minio_virtual_folder}",
        })
        return bundle
