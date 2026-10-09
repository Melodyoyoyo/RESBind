from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import List, Tuple

from .settings import Settings


def checks(settings: Settings, netsurfp_csv: Path = None) -> List[Tuple[str, bool, str]]:
    results = []
    for module in ["torch", "numpy", "pandas"]:
        available = importlib.util.find_spec(module) is not None
        results.append((f"python:{module}", available, "installed" if available else "missing"))
    manifest = settings.bundle_dir / "manifest.json"
    results.append(("v6_bundle", manifest.exists(), str(manifest)))
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            missing = [name for name in data["files"].values() if not (settings.bundle_dir / name).exists()]
            results.append(("v6_bundle_files", not missing, "ok" if not missing else "missing: " + ", ".join(missing)))
        except Exception as exc:
            results.append(("v6_bundle_manifest", False, repr(exc)))
    if netsurfp_csv is not None:
        exists = Path(netsurfp_csv).is_file()
        results.append(("netsurfp", exists, "User CSV; content/sequence validation occurs during predict: " + str(netsurfp_csv)))
    elif settings.netsurfp_command:
        executable = shutil.which(settings.netsurfp_command[0]) or Path(settings.netsurfp_command[0]).exists()
        results.append(("netsurfp", bool(executable), settings.netsurfp_command[0]))
    else:
        results.append(("netsurfp", False, "RESBIND_NETSURFP_COMMAND_JSON is not set"))
    for label, directory in [("job_root", settings.job_root), ("cache_root", settings.cache_root)]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            results.append((label, True, str(directory)))
        except OSError as exc:
            results.append((label, False, repr(exc)))
    try:
        import torch
        detail = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mode"
        results.append(("compute", True, detail))
    except Exception as exc:
        results.append(("compute", False, repr(exc)))
    return results


def run_doctor(settings: Settings, netsurfp_csv: Path = None) -> bool:
    results = checks(settings, netsurfp_csv=netsurfp_csv)
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return all(passed for _, passed, _ in results)

