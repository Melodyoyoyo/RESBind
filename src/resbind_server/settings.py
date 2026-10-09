from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


WEBSERVER_ROOT = Path.cwd()


@dataclass(frozen=True)
class Settings:
    bundle_dir: Path = WEBSERVER_ROOT / "models" / "v6_bidirectional_seed42"
    job_root: Path = WEBSERVER_ROOT / "runtime" / "jobs"
    cache_root: Path = WEBSERVER_ROOT / "runtime" / "cache"
    device: str = "auto"
    max_sequence_length: int = 1024
    esmfold_chunk_size: int = 64
    netsurfp_timeout_seconds: int = 1800
    netsurfp_command: Optional[Tuple[str, ...]] = None
    host: str = "127.0.0.1"
    port: int = 7860

    @classmethod
    def from_env(cls) -> "Settings":
        raw_command = os.environ.get("RESBIND_NETSURFP_COMMAND_JSON", "").strip()
        command = None
        if raw_command:
            parsed = json.loads(raw_command)
            if not isinstance(parsed, list) or not parsed or not all(isinstance(value, str) for value in parsed):
                raise ValueError("RESBIND_NETSURFP_COMMAND_JSON must be a non-empty JSON array of strings.")
            command = tuple(parsed)
        return cls(
            bundle_dir=Path(os.environ.get("RESBIND_BUNDLE_DIR", cls.bundle_dir)),
            job_root=Path(os.environ.get("RESBIND_JOB_ROOT", cls.job_root)),
            cache_root=Path(os.environ.get("RESBIND_CACHE_ROOT", cls.cache_root)),
            device=os.environ.get("RESBIND_DEVICE", "auto"),
            netsurfp_command=command,
            host=os.environ.get("RESBIND_HOST", "127.0.0.1"),
            port=int(os.environ.get("RESBIND_PORT", "7860")),
        )

