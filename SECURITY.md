# Security Policy

## Supported Versions

Only the latest tagged release receives security fixes. Users are strongly
encouraged to upgrade to the newest release.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| older   | :x:                |

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities. Instead,
report them privately by email to:

- **Chen Zhang** — `chen2.zhang@szhittech.com`

Please include:

- The affected version(s) and Python version.
- A description of the vulnerability and its potential impact.
- A minimal reproduction, if possible.

You should receive an acknowledgement within 5 business days. We will keep you
informed of the progress towards a fix and release. Once a fix is available, a
security advisory will be published in the repository.

## Security considerations for this project

het-ai runs `dvc` CLI subprocesses and makes authenticated HTTP calls to
GitHub and object storage (SeaweedFS/S3). Keep the following in mind:

- **Secrets**: `DVC_GITHUB_TOKEN`, `SEAWEEDFS_ACCESS_KEY`, and
  `SEAWEEDFS_SECRET_KEY` should be provided via environment variables or a
  secret manager, never committed to the repository.
- **Supply chain**: verify tagged releases (`pip install het-ai==<version>`)
  and the corresponding PyPI provenance.
- **Untrusted data**: when loading data from external sources in `load_data()`,
  treat it as untrusted input.
