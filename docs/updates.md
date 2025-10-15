## 10/15

Found a large QUIC network traffic dataset **CESNET-QUIC22**. Paper can be found at https://doi.org/10.1016/j.dib.2023.108888, it describes the dataset structure and experimental methods regarding data collection, filtration, sampling, etc. Dataset is available at https://zenodo.org/records/10728760 and can be downloaded using:

`wget https://zenodo.org/api/records/10728760/files/cesnet-quic22.zip/content -O cesnet-quic22.zip`

The authors also maintain the [CESNET DataZoo Python library](https://github.com/CESNET/cesnet-datazoo/tree/main) which provides tools for working with large network traffic datasets. It lets you initialize a dataset to create train, validation, and test dataframes. This could be a viable option if used minimally without reducing the scope of the project.

However, a **key limitation** of this dataset is that it only contains QUIC traffic. One of the questions in the research proposal asks *How accurately can the model distinguish between QUIC and non-QUIC traffic?*, which will require training the model on non-QUIC traffic as well. At the current stage, it is best to proceed with only the QUIC dataset but the project can be scaled up to train on and identify non-QUIC traffic if time permits.