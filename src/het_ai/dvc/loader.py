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
    Strategy interface that decides which data version should be used.

    The framework provides GitHubTagResolver (default); users may implement
    their own:

    - ``SemverLatestResolver`` — pick the latest version by semantic versioning
    - ``GitLabTagResolver`` — integrate with the GitLab API
    - ``FixedTagResolver`` — fix a version (offline / reproduction scenarios)
    - ``EnvTagResolver`` — read the tag from an environment variable (CI)
    """
    def resolve(self) -> Tuple[str, str]:
        """
        Returns (tag_name, commit_sha).
        tag_name is written to DataBundle.meta["dvc_version"] as the unique
        identifier of the data version.
        """
        ...


class GitHubTagResolver:
    """
    Resolves the data version tag from a GitHub repository.
    Supports two strategies (controlled by ``DVCConfig.tag_strategy``):

    - ``"release"`` — take the newest tag whose name matches ``release*`` (the
      current behavior of ``mlops_demo``).
    - ``"latest"`` — take the newest tag by date (regardless of name).
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
    Fixed version resolver: uses the specified tag directly instead of resolving
    the latest version through the GitHub API.

    Note: this resolver only skips the tag resolution step; DVCLoader.pull()
    still needs network access to GitHub to download the .dvc pointer files and
    pull the actual data from SeaweedFS (S3). For fully offline runs, skip
    pull() and use locally cached data instead.

    Use cases:

    - Reproducing an experiment for a specific version (tag known, no dynamic
      resolution needed)
    - CI where the data version is decided by an external system
    - Local debugging that avoids GitHub API calls for tag resolution
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
    Distills the capabilities of ``mlops_demo.GitHubDVCLoader`` into a native
    framework component.

    Design split:

    - ``TagResolver`` — decides "which version to use" (replaceable; defaults
      to :class:`GitHubTagResolver`).
    - ``DVCLoader`` — decides "how to get the data" (fixed DVC operations).

    Basic usage (zero configuration, reads from environment variables)::

        loader = DVCLoader(DVCConfig())
        tag, sha = loader.pull(Path("dvc_data"))
        bundle = DataBundle(...)
        return loader.enrich_bundle(bundle, tag, sha)

    Custom version resolution strategy::

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
        Full data pull workflow:
          1. Determine the data version tag via TagResolver
          2. Download the .dvc pointer files from GitHub
          3. Configure the DVC remote (SeaweedFS S3), run dvc pull + checkout
          4. Return (tag, commit_sha) for use by enrich_bundle
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
        """Inject DVC version metadata into DataBundle.meta to close the data->experiment traceability loop."""
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
