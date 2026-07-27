# Security Policy

## Secret scanning

This repository uses Gitleaks in two places:

- `.pre-commit-config.yaml` checks staged changes before a local commit.
- `.github/workflows/secret-scan.yml` checks the current tracked tree on pull requests and pushes to `main`.

Install the local hook once per clone:

```bash
pip install pre-commit
pre-commit install
```

Never commit real API keys, access tokens, private keys, exported application data, chat history, parsed documents, or test output containing production data. Keep credentials in an untracked `.env` file and use explicit placeholder values in examples and fixtures.

## Historical exposure

Secret scanning of the current tree does not remove data from Git history. When a credential or personal document has appeared in history, rotate the credential immediately and perform a coordinated history rewrite before enabling full-history scanning.

## Reporting

Do not open a public issue containing a secret. Report suspected exposure privately to the repository maintainers, including the affected path and commit hash but never the secret value.
