# Naming migration — 2026-10-09

The active workflow uses semantic names documented in
[EXPERIMENTS.md](../../EXPERIMENTS.md). This directory preserves the pre-rename
source files and old reproduction guide. It is historical material, not an
active workflow or test suite.

`manifest.json` records original source hashes, source moves, and checksums for
each existing output that was renamed. Figure bytes were preserved. Missing
computational results were not reconstructed from manuscript PDFs.

Existing simulation manifests are unchanged: their old paths and hashes describe
the code actually used for those runs. In particular, the original retraining
module and the nuisance modules that imported it are preserved under `sources/`.
Other unchanged dependencies remain in the repository and Git history. These
snapshots are not a complete standalone environment.

Renaming imports changes source hashes even though the statistical code is
unchanged. Existing checkpoint compatibility checks therefore still reject
mixing old-source shards with a new-source run. Use the original source snapshot
to resume an old run, or a fresh output directory for a new run; do not rewrite
old manifests to make a mismatch disappear. Saved summaries can be plotted under
the new names without rerunning simulations.
