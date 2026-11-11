# Pre-release audit — 0.1.0

## Checks completed

- The repository contains 157 files: 156 payload files and `SHA256SUMS.txt`.
- All 38 Python files passed Python's AST syntax check.
- A text-file scan found no machine-specific absolute paths tied to the original Linux or macOS user directories.
- A text-file scan found no matches for common GitHub personal-access-token, AWS access-key, private-key, or generic API-key patterns.
- No WAM checkpoint, LaMa checkpoint, COCO image archive, or DIV2K image archive is included.
- The 156 entries in `SHA256SUMS.txt` were verified against the package files.
- GitHub repository metadata reports the repository visibility as Public.
- The LaMa Smoke5 record preserves both launch attempts. `run.log` records the initial import failure; `run_retry.log` records the successful retry, in which five images were processed and 30 strategy-image rows were written.
- The Git blob SHA-1 identifiers of 40 selected WAM files match the corresponding files at upstream commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e`. The file-by-file results are in `protocols/wam_upstream_fingerprint_audit.csv`.

## Checks still required before public release

- Open the repository and the required source/configuration files in a browser session that is not signed in to GitHub. This logged-out access check has not yet been independently confirmed.
- A clean-machine, end-to-end run of all experiments has not been performed as part of this packaging work.
- Users must install the upstream WAM and LaMa dependencies and obtain the required checkpoints and datasets separately.
- Review the applicable third-party licenses and confirm that the included result records and logs are suitable for public distribution.

## Scope of this audit

The WAM comparison covers 40 selected files only. It does not establish that every file in the original experiment directory is identical to upstream. The original experiment directory did not retain Git metadata, so its complete historical checkout cannot be recovered.

The checks listed above are static packaging, syntax, path, and checksum checks. They do not establish that every experiment can be reproduced end-to-end on an arbitrary machine. The repository should therefore be described as a code and experiment-record package with documented dependencies, not as an independently revalidated full reproduction.
