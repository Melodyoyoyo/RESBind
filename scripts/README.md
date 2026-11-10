# Repository utilities

## Refresh the file manifest

After committing file changes, run the following command from a local checkout:

```sh
python scripts/update_release_manifest.py
```

The script updates the checksums of tracked files in `release_manifest.json` and checks that model artifacts match their recorded checksums. Review and commit the updated manifest alongside the repository changes.
