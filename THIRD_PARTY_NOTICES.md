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
not redistribution rights. Publication clearance and any required attribution for
these example inputs remain to be confirmed before public release, particularly
whether the exported NetSurfP CSV may accompany the public software example.

### Suggested clarification request to NetSurfP

Subject: Redistribution of a NetSurfP-3.0 prediction CSV as a research example

We are preparing RESBind, a research tool for protein binding-residue prediction,
for public release. We would like to include one CSV containing NetSurfP-3.0 web
server predictions for the public UniProt sequence P07014, so users can test our
downstream inference without submitting a new job. We would cite NetSurfP-3.0
and link to your official service. We would not distribute NetSurfP software or
weights, or claim that our MIT license grants rights in your materials. Is public
redistribution of this prediction CSV permitted, and is any additional attribution
or restriction required? Our own code and model will allow commercial use under
MIT; please clarify separately whether conditions apply to the bundled CSV.

This is an unsent draft, not evidence of permission or provider approval.

Python dependencies retain their respective licenses. Installing RESBind does not
replace those licenses with MIT.
