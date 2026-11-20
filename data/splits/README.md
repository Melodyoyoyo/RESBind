# Benchmark dataset splits

Protein identifiers for the RESBind benchmark, with one UniProt accession per line.

| File | Split | Proteins |
| --- | --- | ---: |
| `train_ids.txt` | Training | 805 |
| `validation_ids.txt` | Validation | 202 |
| `test_ids.txt` | Test | 300 |

The lists correspond to the proteins retained in the training and evaluation feature tables used for the study. The three sets contain no shared protein identifiers.

Protein sequences and binding-residue annotations originate from the [bindPredict dataset](https://github.com/Rostlab/bindPredict), described by [Littmann et al. (2021)](https://doi.org/10.1038/s41598-021-03431-4). RESBind retains the original development/test partition; the development set is divided into training and validation subsets as described in the manuscript Methods.
