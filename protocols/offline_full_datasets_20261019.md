# Offline full-source data handoff, before preparation

User explicitly requests all WfInstances data locally downloaded, adapted with
communication/capacity/hard-budget constraints, and pushed into ordinary Git so
a company computer can use it without external downloads. A subsequent steering
message adds TPC-H in parallel. No new model training or test-based method
selection is authorized by this data protocol.

Pin WfInstances d7bfead08a4dbf985ebbf427b3685a6512c259e2: complete recursive
tree has318JSON/351blobs,892484319JSONbytes. Preserve every source file, original
Git blob SHA and SHA256, provenance/readmes/license. Raw downloads now include
previously reserved TEST paths by explicit user instruction; downloading is not
model evaluation, and they must not become training targets automatically.

TPC-H means research query DAGs from the PPO-BiHyb author data, rather than
claiming to ship a complete certified TPC-H database benchmark. Audit local
author checkout dbb9a1a2613d8e306fa726089e2c294a29f9f133 and public source before
conversion; preserve original inputs and document missing communication or
hardware attributes as simulation assumptions. Final source count determined
from file inventory, not guessed here.

Adaptation initially follows current semantics: nonpreemptive tasks, fixed
overlapping communication with no link contention, cumulative pool resources,
additive hard execution-cost budget. Default3/6pools,2networkprofiles,2budget
etas(.25/.6). Retain measured runtimes/file sizes where supplied. Record source
topology separately from platform variants. All workflow sizes are preserved;
large graphs require sparse storage rather than allocating dense N-by-N data.

Ship compressed source files and sparse adapted records as normal Git blobs,
each below GitHub's100MiB limit (target below50MiB), with deterministic hashes.
Do not depend on Git LFS, releases, model hubs or first-run external downloads.
Provide stdlib-only offline integrity/extraction tools and a bridge to existing
training schema for tractable graphs. Record conversion failures explicitly and
resolve them; no silently skipped source.

Do not generate TRAIN labels for an undivided archive. Provide grouped split
instructions separately, retain prior seen-source flags and author split
metadata where present, and avoid claiming a fresh final TEST from this handoff.
Only TRAIN selected by an explicit split may later acquire teacher labels.
Validate source completeness, sparse physical fields, DAG acyclicity, budgets,
conversion compatibility with old public pilot, materialization/decoder checks
on a bounded representative engineering subset, and an offline unpack roundtrip.
These checks are not model quality results. Keep frozen106files and previous
experiment evidence unchanged.
