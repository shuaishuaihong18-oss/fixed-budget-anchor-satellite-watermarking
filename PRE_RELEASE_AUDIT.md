# Pre-release audit — 0.1.0

## Checks completed

- Archive contains 155 files (source scripts, configurations, manifests, logs, raw records, summaries, and documentation).
- All 38 Python files parse successfully with Python's AST parser.
- Text-file scan found no machine-specific absolute paths tied to the original Linux or macOS user directories.
- Text-file scan found no matches for common GitHub personal-access-token, AWS access-key, private-key, or generic API-key patterns.
- No WAM checkpoint, LaMa checkpoint, COCO image archive, or DIV2K image archive is included.
- SHA-256 checksums are provided in `SHA256SUMS.txt`.

## Checks not yet completed

- A clean-machine, end-to-end execution of all experiments has not been performed as part of this packaging pass.
- Third-party WAM and LaMa installation, checkpoint retrieval, and dataset download must be completed by users according to upstream instructions.
- Before public release, the authors should review third-party licenses and confirm that all included result records and logs are suitable for public distribution.

## Interpretation

The checks above are static packaging and syntax checks. They do not establish that every experiment can be reproduced end-to-end on an arbitrary machine. The release should therefore be described as a reproducibility package with documented dependencies and archived experiment records, not as an independently revalidated full reproduction.
