# Pre-release audit — 0.1.0

## Checks completed

- The archive contains 157 files in total: 156 payload files plus `SHA256SUMS.txt`.
- All 38 Python files parse successfully with Python's AST parser.
- Text-file scan found no machine-specific absolute paths tied to the original Linux or macOS user directories.
- Text-file scan found no matches for common GitHub personal-access-token, AWS access-key, private-key, or generic API-key patterns.
- No WAM checkpoint, LaMa checkpoint, COCO image archive, or DIV2K image archive is included.
- SHA-256 checksums are provided in `SHA256SUMS.txt`.
- The LaMa Smoke5 execution record preserves both attempts: `run.log` documents the initial import failure, while `run_retry.log` documents the successful five-image, 30-row retry.

- The Git blob SHA-1 identifiers of 40 selected WAM files match the official upstream commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e`; the file-by-file record is in `protocols/wam_upstream_fingerprint_audit.csv`. This does not establish that every file in the experiment directory is identical to upstream, because the original experiment directory did not retain Git metadata.

- The Git blob SHA-1 identifiers of 40 selected WAM files match the official upstream commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e`; the file-by-file record is in `protocols/wam_upstream_fingerprint_audit.csv`. This does not establish that every file in the experiment directory is identical to upstream, because the original experiment directory did not retain Git metadata.

## Checks not yet completed

- A clean-machine, end-to-end execution of all experiments has not been performed as part of this packaging pass.
- Third-party WAM and LaMa installation, checkpoint retrieval, and dataset download must be completed by users according to upstream instructions.
- Before public release, the authors should review third-party licenses and confirm that all included result records and logs are suitable for public distribution.

## Interpretation

The checks above are static packaging and syntax checks. They do not establish that every experiment can be reproduced end-to-end on an arbitrary machine. The release should therefore be described as a reproducibility package with documented dependencies and archived experiment records, not as an independently revalidated full reproduction.
