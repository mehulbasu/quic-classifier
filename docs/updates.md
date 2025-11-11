## 10/15

Found a large QUIC network traffic dataset **CESNET-QUIC22**. Paper can be found at https://doi.org/10.1016/j.dib.2023.108888, it describes the dataset structure and experimental methods regarding data collection, filtration, sampling, etc. Dataset is available at https://zenodo.org/records/10728760 and can be downloaded using:

`wget https://zenodo.org/api/records/10728760/files/cesnet-quic22.zip/content -O cesnet-quic22.zip`

The authors also maintain the [CESNET DataZoo Python library](https://github.com/CESNET/cesnet-datazoo/tree/main) which provides tools for working with large network traffic datasets. It lets you initialize a dataset to create train, validation, and test dataframes. This could be a viable option if used minimally without reducing the scope of the project.

However, a **key limitation** of this dataset is that it only contains QUIC traffic. One of the questions in the research proposal asks *How accurately can the model distinguish between QUIC and non-QUIC traffic?*, which will require training the model on non-QUIC traffic as well. At the current stage, it is best to proceed with only the QUIC dataset but the project can be scaled up to train on and identify non-QUIC traffic if time permits.

## 11/2
`midterm-report.md` contains progress.

## 11/10

Kicked off the distributed XGBoost effort by wiring `train_xgboost.py` to spin up a multi-GPU Dask cluster, mirror the RAPIDS feature-engineering flow, and stream QUIC parquet shards into the training graph. The first version used `DaskXGBClassifier` because of its sklearn-style API, but the approach forced each worker to materialize dense gradient buffers for all 44M rows × 105 classes. Every run hit 8–9 GB allocations per worker and ultimately ran out of device memory.

Resolved the memory wall by redesigning the training loop around `xgboost.dask.train` with `DaskQuantileDMatrix`. The quantile matrix keeps only histograms/weights on each GPU and streams batches during boosting, dropping per-worker usage well below the 14 GB limit. After that change, the four V100s finished a 10-round test in ~47 s and the full 400-round experiment in ~27 min without OOMs.

While stress-testing the pipeline we also tightened the evaluation path, swapping the earlier label-map compute for a single-pass categorical encode and ensuring predictions stayed 2-D for the 105-class argmax. This cut a redundant dataset scan and kept the class index stable between train/eval splits.

Partition management was another major lever. Initially we repartitioned the raw parquet input by `partition_size`, but once the GPU feature engineering exploded each row into hundreds of histogram features, feature and label collections drifted out of sync. The workflow now delays repartitioning until after feature creation and uses a shared `npartitions` target (the CLI flag I switched from `partition_size`). This keeps features/labels aligned and has been stable across all subsequent runs.

Recent large-scale benchmarks (see `results.txt`) show the evolution: the first stable 400-round run reached 0.7026 accuracy / 0.6147 macro-F1 in ~27 min; tuning depth/bin parameters produced up to 0.7061 accuracy / 0.6147 macro-F1 on the 8 M-flow holdout. Training logs also confirm the balanced partitions (~441k rows each after repartitioning 4 files) and consistent evaluation throughput.

Finally, we hardened shutdown by suppressing benign Dask heartbeat errors and guarding the client/cluster teardown, so long runs exit cleanly after writing the booster to `datasets/cache/models/xgboost_quic.json`.
