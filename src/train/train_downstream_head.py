from __future__ import annotations

# Within-only compatibility shim.
# The rest of the codebase still imports `train.train_downstream_head`;
# re-export the standardized within implementation so existing entrypoints
# keep working after the non-within training file was archived.
from .within_train_downstream_head import *  # noqa: F401,F403
