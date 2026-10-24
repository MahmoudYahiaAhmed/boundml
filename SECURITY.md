# Security policy

## Supported versions

BoundML is an early release (0.x). Only the latest released version receives fixes.

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately through this repository's **Security → Report a vulnerability**
form (GitHub private vulnerability reporting). Only maintainers can see these
reports. **While the repository is private this form is not available:** GitHub offers
it only on public repositories, and the maintainer will enable it when the repository
is made public. Until then, report to the maintainer through a private channel you
already share; do not open an issue. Include:

- the BoundML version (`boundml version`) and platform;
- a minimal package, CSV or HTTP request that reproduces the problem;
- the impact you observed (crash, excessive memory, wrong result, file write outside
  the expected path, ...).

You should receive an acknowledgement within 7 days. Fixes are released as patch
versions and credited in `CHANGELOG.md` unless you prefer otherwise.

## Scope and trust model

In scope: anything that lets a crafted **package, checkpoint, CSV file, import bundle
or HTTP request** crash the process, exceed the documented resource bounds, read or
write files other than those named on the command line, or silently produce wrong
predictions.

Known limits that are documented and not vulnerabilities by themselves:

- SHA-256 checksums detect corruption, but they do not authenticate a package's
  author. Only load packages from sources you trust.
- `boundml serve` has no authentication or TLS and binds to `127.0.0.1` by default.
  Put a reverse proxy in front of it before exposing it to a network.
- The memory bound covers planned tensor storage only, not whole-process memory (see
  `docs/architecture.md`).
