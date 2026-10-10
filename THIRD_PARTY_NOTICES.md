# License scope and third-party materials

The MIT license in LICENSE applies to the original RESBind code, documentation,
and the RESBind V6 checkpoint and scalers in models/v6_bidirectional_seed42,
to the extent that the copyright holder owns the applicable rights.

It does not relicense third-party software, upstream model weights, source
sequences or third-party-derived example features. Retain applicable upstream
notices and comply with the terms associated with each resource.

## Upstream resources

- ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50 . The official
  [ProtTrans repository](https://github.com/agemagician/ProtTrans#-license)
  identifies its pretrained models as Academic Free License v3.0 (AFL-3.0).
  This statement concerns upstream models, not a new license grant for outputs.
  Reference: Elnaggar et al., https://doi.org/10.1109/TPAMI.2021.3095381 .
- ESMFold: https://huggingface.co/facebook/esmfold_v1 . The official model card
  lists MIT. Structure predictions are not experimental structures.
- NetSurfP: https://services.healthtech.dtu.dk/services/NetSurfP-3.0/ . The official
  instructions support exporting prediction results as CSV, JSON and ZIP. The
  reviewed page did not explicitly address redistribution of exported predictions
  inside another software repository. Do not equate this uncertainty with a ban,
  or apply the standalone software agreement to online outputs without evidence.
  Reference: Hoie et al., https://doi.org/10.1093/nar/gkac439 .
- UniProt: https://www.uniprot.org/help/license . Copyrightable database content
  is covered by [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Attribution: UniProt Consortium, entry
  [P07014](https://www.uniprot.org/uniprotkb/P07014/entry). The example FASTA was
  reconstructed from the saved ESMFold structure; its header differs from the
  database FASTA. An unchanged current UniProt sequence has not been asserted.

These official pages were reviewed on 2026-10-10. License labels for upstream
code/models are not, by themselves, a determination of the rights in every output.

Upstream ProtT5/ESMFold weights and NetSurfP software are not bundled here.
The Skill guides users to obtain required resources separately. Their availability
does not imply endorsement of RESBind by their providers.

## Bundled example inputs

The P07014 example includes a sequence, ESMFold structure, ProtT5 embeddings and
NetSurfP results. These files are not granted a new MIT license by this repository.
Their file hashes are recorded in release_manifest.json; hashes establish identity,
not redistribution rights. The NetSurfP CSV is a saved prediction result, not
NetSurfP software or model weights. No special redistribution permission from
DTU has been obtained or claimed; the online-output terms remain as described
above. RESBind's MIT license does not override applicable third-party terms.

Python dependencies retain their respective licenses. Installing RESBind does not
replace those licenses with MIT.
