"""Run against the installed package with the official SDK stdio client."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pandas as pd


async def run(root):
    env = os.environ.copy()
    env.update(RESBIND_MCP_WORKSPACE=str(root),
               RESBIND_BUNDLE_DIR=str(root / "models/v6_bidirectional_seed42"))
    params = StdioServerParameters(command=sys.executable,
                                   args=["-I", "-m", "resbind_server.mcp_server"], env=env)
    args = {"fasta": "examples/P07014/sequence.fasta", "pdb": "examples/P07014/structure.pdb",
            "netsurfp_csv": "examples/P07014/netsurfp.csv", "prott5_npz": "examples/P07014/prott5.npz"}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=210)) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert names == {"check_environment", "inspect_fasta", "predict_saved"}, names
            result = await session.call_tool("check_environment", {})
            assert not result.isError, result
            for invalid in ["../outside.fasta", "missing.fasta"]:
                print("Checking rejection:", invalid, flush=True)
                result = await session.call_tool("inspect_fasta", {"fasta": invalid})
                assert result.isError
            print("Checking saved inputs", flush=True)
            result = await session.call_tool("inspect_fasta", args)
            assert not result.isError, result
            info = json.loads(result.content[0].text)
            assert info["ready_for_saved_prediction"] and info["length"] == 238
            result = await session.call_tool("inspect_fasta", {**args, "mode": "train"})
            assert not result.isError, result
            assert json.loads(result.content[0].text)["labels_required"]
            print("Running CPU prediction", flush=True)
            result = await session.call_tool("predict_saved", args)
            assert not result.isError, result
            output = json.loads(result.content[0].text)
    actual = pd.read_csv(output["artifacts"]["residue_predictions.csv"])
    summary = output['summary']
    assert summary['pipeline_schema'] == 3
    candidates = pd.read_csv(output['artifacts']['pockets.csv'])
    assert {'task','candidate_type','threshold','cutoff_angstrom'} <= set(candidates.columns)
    for task in ['metal','nucleic_acid','small_molecule']:
        groups = summary['pockets_by_task'][task]
        recovered = sorted(r for p in groups for r in p['residues'])
        expected_positions = actual.loc[actual['pred_'+task],'position'].tolist()
        assert recovered == expected_positions
        assert summary['pocket_counts'][task] == sum(len(p['residues'])>1 for p in groups)
        assert summary['isolated_residue_counts'][task] == sum(len(p['residues'])==1 for p in groups)
        assert len(candidates[candidates.task==task]) == len(groups)
    expected = pd.read_csv(root / "examples/P07014/expected_predictions.csv")
    assert actual[["protein_id", "position", "amino_acid"]].equals(expected[["protein_id", "position", "amino_acid"]])
    deltas = {}
    for task in ["metal", "nucleic_acid", "small_molecule"]:
        delta = float((actual["prob_"+task]-expected["prob_"+task]).abs().max())
        assert delta < 1e-6
        assert actual["pred_"+task].equals(expected["pred_"+task])
        deltas[task] = delta
    import importlib.metadata
    report = {"passed": True, "transport": "stdio", "sdk_version": importlib.metadata.version("mcp"),
              "tools": sorted(names), "max_score_differences": deltas,
              "identical_calls": True, "path_rejection_passed": True,
              "pipeline_schema": 3, "three_task_candidate_checks_passed": True,
              "checkpoint_sha256": hashlib.sha256((root / "models/v6_bidirectional_seed42/model.pt").read_bytes()).hexdigest(),
              "scope": "Installed package, saved P07014 features, CPU; not full new-protein or independent Skill acceptance."}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run(args.repository.resolve()))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
