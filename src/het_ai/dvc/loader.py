from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path
from typing import Protocol, Tuple, runtime_checkable

import requests

from het_ai.dvc.config import DVCConfig
from het_ai.studio.bundle import DataBundle


# ──────────────────────────────────────────────────────────────────────────────
# 版本解析协议：唯一的解耦点
# ──────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class TagResolver(Protocol):
    """
    决定"当前应该使用哪个数据版本"的策略接口。

    框架提供 GitHubTagResolver（默认），用户可自行实现：
      - SemverLatestResolver   按语义化版本取最新
      - GitLabTagResolver      对接 GitLab API
      - FixedTagResolver       固定版本（离线/复现场景）
      - EnvTagResolver         从环境变量读取 tag（CI 场景）
    """
    def resolve(self) -> Tuple[str, str]:
        """
        返回 (tag_name, commit_sha)。
        tag_name 将写入 DataBundle.meta["dvc_version"]，作为数据版本的唯一标识。
        """
        ...


class GitHubTagResolver:
    """
    从 GitHub 仓库解析数据版本 tag。
    支持两种策略（由 DVCConfig.tag_strategy 控制）：

      "release" — 取名称匹配 release* 的最新 tag（mlops_demo 的当前行为）
      "latest"  — 取时间最新的 tag（无论名称）
    """

    def __init__(self, config: DVCConfig):
        self._config = config
        self._api_base = config.github_api_base.rstrip("/")
        self._owner_repo = self._parse_owner_repo(config.github_repo)
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"token {config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "het-ai/DVCLoader",
        })

    @staticmethod
    def _parse_owner_repo(github_repo: str) -> str:
        if github_repo.startswith("https://"):
            m = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", github_repo)
            if not m:
                raise ValueError(f"无效的 GitHub 仓库 URL: {github_repo}")
            return m.group(1)
        if "/" not in github_repo:
            raise ValueError(f"仓库格式应为 'owner/repo'，收到: {github_repo}")
        return github_repo

    def _api(self, endpoint: str, **kwargs):
        url = f"{self._api_base}/{endpoint.lstrip('/')}"
        resp = self._session.get(url, **kwargs)
        resp.raise_for_status()
        # 自动处理 GitHub 分页
        data = resp.json()
        while "next" in resp.links:
            resp = self._session.get(resp.links["next"]["url"])
            resp.raise_for_status()
            data.extend(resp.json())
        return data

    def resolve(self) -> Tuple[str, str]:
        tags = self._api(f"/repos/{self._owner_repo}/tags")

        if not tags:
            # 没有 tag，退化到最新 commit
            commits = self._api(
                f"/repos/{self._owner_repo}/commits", params={"per_page": 1}
            )
            if not commits:
                raise RuntimeError(f"仓库 {self._owner_repo} 没有任何提交")
            return "latest", commits[0]["sha"]

        strategy = self._config.tag_strategy

        if strategy == "release":
            for tag in tags:
                if re.match(r"release", tag["name"], re.I):
                    sha = tag.get("commit", {}).get("sha", "")
                    return tag["name"], sha
            raise RuntimeError(
                f"仓库 {self._owner_repo} 中找不到 release* tag，"
                f"可将 tag_strategy 改为 'latest' 使用最新 tag。"
            )

        if strategy == "latest":
            # GitHub Tags API 的 commit 对象只是轻量指针，不含 committer.date 字段。
            # 需要调用 Commits API 补全每个 tag 的实际提交时间，再取最新。
            def _fetch_commit_date(tag) -> str:
                sha = tag.get("commit", {}).get("sha", "")
                if not sha:
                    return ""
                try:
                    commit = self._api(f"/repos/{self._owner_repo}/commits/{sha}")
                    return (
                        commit.get("commit", {})
                        .get("committer", {})
                        .get("date", "")
                    )
                except Exception:
                    return ""

            best = max(tags, key=_fetch_commit_date)
            return best["name"], best.get("commit", {}).get("sha", "")

        raise ValueError(
            f"未知的 tag_strategy='{strategy}'，支持 'release' 或 'latest'。"
        )


class FixedTagResolver:
    """
    固定版本解析器：直接使用指定的 tag，不通过 GitHub API 解析最新版本。

    注意：此 Resolver 仅跳过 tag 解析步骤，DVCLoader.pull() 仍需网络访问
    GitHub 下载 .dvc 指针文件并从 SeaweedFS(S3) 拉取实际数据。
    若需完全离线运行，应跳过 pull() 直接使用已缓存的本地数据。

    适用场景：
      - 复现指定版本的实验（tag 已知，无需动态解析）
      - CI 中由外部系统决定数据版本
      - 本地调试时避免 tag 解析的 GitHub API 调用 
    """

    def __init__(self, tag: str, commit_sha: str = ""):
        self._tag = tag
        self._sha = commit_sha

    def resolve(self) -> Tuple[str, str]:
        return self._tag, self._sha


# ──────────────────────────────────────────────────────────────────────────────
# DVCLoader 主体：DVC 机械操作，与 Git 提供商无关
# ──────────────────────────────────────────────────────────────────────────────

class DVCLoader:
    """
    将 mlops_demo.GitHubDVCLoader 的能力提炼为框架原生组件。

    设计分工：
      TagResolver  — 决定"用哪个版本"（可替换，默认 GitHubTagResolver）
      DVCLoader    — 决定"怎么拿数据"（DVC 机械操作，固定不变）

    基本用法（零配置，从环境变量读取）：
        loader = DVCLoader(DVCConfig())
        tag, sha = loader.pull(Path("dvc_data"))
        bundle = DataBundle(...)
        return loader.enrich_bundle(bundle, tag, sha)

    自定义版本解析策略：
        resolver = FixedTagResolver("release-20250101")
        loader = DVCLoader(config, tag_resolver=resolver)
    """

    def __init__(
        self,
        config: DVCConfig,
        tag_resolver: TagResolver | None = None,
    ):
        self.config = config
        self._resolver = tag_resolver or GitHubTagResolver(config)
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"token {config.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "het-ai/DVCLoader",
        })
        # self._owner_repo = GitHubTagResolver._parse_owner_repo(config.github_repo)
        self._api_base = config.github_api_base.rstrip("/")

        # 仅在 github_repo 非空时解析（FixedTagResolver 场景下可能为空）
        self._owner_repo = (
            GitHubTagResolver._parse_owner_repo(config.github_repo)
            if config.github_repo else ""
        )

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def pull(self, output_path: Path) -> Tuple[str, str]:
        """
        完整数据拉取流程：
          1. 通过 TagResolver 确定数据版本 tag
          2. 从 GitHub 下载 .dvc 指针文件
          3. 配置 DVC remote（SeaweedFS S3），执行 dvc pull + checkout
          4. 返回 (tag, commit_sha) 供 enrich_bundle 使用
        """
        tag, commit_sha = self._resolver.resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        dvc_files = self._list_dvc_files(tag)
        if self.config.dvc_pattern:
            dvc_files = [f for f in dvc_files if f.startswith(self.config.dvc_pattern)]

        if not dvc_files:
            raise RuntimeError(
                f"在 tag={tag} 中找不到匹配 '{self.config.dvc_pattern}' 的 .dvc 文件。"
            )

        success, fail = 0, 0
        for dvc_path in dvc_files:
            try:
                local_file = self._download_dvc_pointer(tag, dvc_path, output_path)
                self._dvc_pull(local_file)
                success += 1
            except Exception as exc:
                fail += 1
                # 单文件失败不阻断其他文件，但记录
                import logging
                logging.getLogger(__name__).error(
                    f"[DVCLoader] 处理失败 {dvc_path}: {exc}", exc_info=True
                )

        if fail == len(dvc_files):
            raise RuntimeError("所有 .dvc 文件均拉取失败，请检查 SeaweedFS / GitHub 配置。")

        return tag, commit_sha

    def enrich_bundle(
        self, bundle: DataBundle, tag: str, commit_sha: str
    ) -> DataBundle:
        """将 DVC 版本元数据注入 DataBundle.meta，实现数据→实验溯源闭环。"""
        bundle.meta.update({
            "dvc_version":    tag,
            "dvc_commit_sha": commit_sha,
            "dvc_repo":       self.config.github_repo,
            "dvc_remote": f"s3://{self.config.seaweedfs_bucket}/{self.config.seaweedfs_virtual_folder}".rstrip("/")
        })
        return bundle

    # ── 内部：GitHub .dvc 文件获取 ────────────────────────────────────────────

    def _list_dvc_files(self, tag: str) -> list[str]:
        url = f"{self._api_base}/repos/{self._owner_repo}/git/trees/{tag}"
        resp = self._session.get(url, params={"recursive": "1"})
        resp.raise_for_status()
        return [
            item["path"]
            for item in resp.json().get("tree", [])
            if item["type"] == "blob" and item["path"].endswith(".dvc")
        ]

    def _download_dvc_pointer(
        self, tag: str, dvc_path: str, output_path: Path
    ) -> Path:
        clean = dvc_path.lstrip("/")
        url = f"{self._api_base}/repos/{self._owner_repo}/contents/{clean}"
        meta = self._session.get(url, params={"ref": tag}).json()

        local_file = output_path / Path(clean).name
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # 大文件 / LFS → download_url 直接下载
        if meta.get("encoding") == "none" and "download_url" in meta:
            content = self._session.get(meta["download_url"], timeout=60).content
        elif "content" in meta:
            raw = base64.b64decode(meta["content"])
            # LFS 指针文本也走 download_url
            if raw.startswith(b"version https://git-lfs"):
                content = self._session.get(meta["download_url"], timeout=60).content
            else:
                content = raw
        else:
            raise ValueError(f"无法解析 GitHub 文件响应: {list(meta.keys())}")

        local_file.write_bytes(content)
        return local_file

    # ── 内部：DVC 机械操作（与 Git 提供商无关）────────────────────────────────

    def _dvc_pull(self, dvc_file: Path) -> None:
        """配置 DVC remote 并拉取实际数据到本地。"""
        remote_url = f"s3://{self.config.seaweedfs_bucket}/{self.config.seaweedfs_virtual_folder}".rstrip("/")
        endpoint_url = f"{'https' if self.config.seaweedfs_secure else 'http'}://{self.config.seaweedfs_endpoint}"
        commands = [
            ["dvc", "init", "--no-scm", "-f"],
            ["dvc", "remote", "add", "-d", "seaweedfs", remote_url, "-f"],
            ["dvc", "remote", "modify", "seaweedfs", "endpointurl", endpoint_url],
            ["dvc", "remote", "modify", "seaweedfs", "access_key_id",
             self.config.seaweedfs_access_key],
            ["dvc", "remote", "modify", "seaweedfs", "secret_access_key",
             self.config.seaweedfs_secret_key],
            ["dvc", "pull", str(dvc_file), "--force"],
            ["dvc", "checkout", "--force", "--with-deps"],
        ]
        for cmd in commands:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"DVC 命令失败: {' '.join(cmd)}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
