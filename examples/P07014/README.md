# P07014 example

This example contains a protein sequence and the precomputed features needed to run RESBind on CPU.

| File | Content |
| --- | --- |
| `sequence.fasta` | Input protein sequence |
| `structure.pdb` | ESMFold structure |
| `netsurfp.csv` | NetSurfP surface features |
| `prott5.npz` | ProtT5 residue embeddings |
| `expected_predictions.csv` | Reference residue predictions for this example |

From the repository root, run:

```sh
resbind predict --fasta examples/P07014/sequence.fasta --pdb examples/P07014/structure.pdb --netsurfp-csv examples/P07014/netsurfp.csv --prott5-npz examples/P07014/prott5.npz --device cpu --out-dir results/P07014
```

The output includes predictions for metal ions, nucleic acids, and small molecules, spatial binding-site candidates, and a result ZIP. See the [prediction guide](../../skills/resbind/references/prediction.md) for input and output details.
