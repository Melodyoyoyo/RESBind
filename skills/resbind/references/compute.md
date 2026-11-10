# Compute options

RESBind supports CPU prediction from prepared features and CUDA-based generation of ProtT5 and ESMFold features.

| Available inputs and resources | Workflow |
| --- | --- |
| Matching FASTA, PDB, NetSurfP CSV, and ProtT5 NPZ | Run CLI or MCP prediction on CPU |
| Missing ProtT5 or ESMFold features, with a local CUDA GPU | Generate features locally, then run prediction |
| A remote CUDA server | Generate features on the server and run prediction locally or remotely |
| A scheduler-managed GPU cluster | Submit feature generation through the cluster scheduler |

## Prepare a CUDA environment

Check available GPU memory, RAM, disk space, and PyTorch CUDA support on the execution machine. Use a dedicated environment for feature generation and obtain the official ProtT5 and ESMFold model snapshots.

Follow [prediction.md](prediction.md) to run ProtT5 and ESMFold sequentially in separate processes. Choose resources appropriate to the protein length and retain completed feature files for reuse.

## Remote execution

Use the user's authorized connection method, project directory, and GPU allocation. Follow the server's resource-sharing policy. For a scheduled cluster, see [hpc.md](hpc.md).

Transfer the input FASTA to the selected machine and save each run in its own directory. Retrieve the ProtT5 NPZ, ESMFold PDB, and provenance records for CPU prediction. Preserve job or process identifiers and logs so interrupted sessions can resume from the same run.

For paid resources, agree the resource and cost limits before provisioning. Keep credentials and machine-specific connection details in local configuration.
