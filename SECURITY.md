# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability in LocalVectorDB, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email **thomas.villani@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes (if applicable)

We will acknowledge receipt within 48 hours and aim to provide a resolution timeline within 5 business days.

## Security Practices

This project uses the following tools to maintain code security:

- **Bandit** - Static analysis for common Python security issues
- **CodeQL** - Automated semantic code analysis via GitHub Actions
- **Ruff** - Linting rules that catch potential security issues
- **Hardened file extraction** - Uploads are converted with remote fetching and
  local-file access disabled, HTML scripts stripped, and file-size / ZIP-bomb
  guards applied before parsing (see the ``[extraction]`` config section)
- **Pre-commit hooks** - Automated checks before code is committed

## Scope

The following are in scope for security reports:

- SQL injection in query builder or metadata filtering
- Path traversal in file upload/extraction endpoints
- Authentication bypass in API key handling
- Arbitrary code execution via embedding provider plugins
- Denial of service via crafted documents or queries
- Information disclosure through error messages or logs

## Out of Scope

- Vulnerabilities in third-party dependencies (report to the upstream project)
- Issues requiring physical access to the server
- Social engineering attacks
