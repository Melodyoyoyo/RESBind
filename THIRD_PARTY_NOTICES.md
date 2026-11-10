# Third-party notices

## RESBind license

The [MIT License](LICENSE) covers the original RESBind code and documentation, and the bundled seed-42 checkpoint and scalers to the extent owned by the copyright holder. Third-party software, model weights, sequences, and derived example features retain their applicable terms.

## Upstream resources

| Resource | Source and attribution |
| --- | --- |
| ProtT5 | [Model](https://huggingface.co/Rostlab/prot_t5_xl_uniref50), [ProtTrans license information](https://github.com/agemagician/ProtTrans#-license), [Elnaggar et al.](https://doi.org/10.1109/TPAMI.2021.3095381) |
| ESMFold | [Model and license information](https://huggingface.co/facebook/esmfold_v1) |
| NetSurfP | [Official service](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/), [Høie et al.](https://doi.org/10.1093/nar/gkac439) |
| UniProt | [Database license](https://www.uniprot.org/help/license), [P07014 entry](https://www.uniprot.org/uniprotkb/P07014/entry), UniProt Consortium |

Obtain upstream model weights and software from their official providers and retain the associated license notices. Python dependencies retain their respective licenses.

## Example inputs

The P07014 example contains a sequence reconstructed from its saved ESMFold structure, ProtT5 embeddings, and NetSurfP prediction results. File checksums are listed in `release_manifest.json`. These third-party-derived inputs are subject to their source terms rather than a new MIT license grant. For redistribution of NetSurfP outputs, consult the service's current terms or contact the provider.
