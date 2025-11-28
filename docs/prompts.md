Read the provided dataset.md and proposal.md files to understand the data we are working with and the scope of the project. I want to be able to train a baseline decision tree model by the end of the day. Currently I have `cesnet-quic22.zip` downloaded. I'm thinking something like first training a model using only one day's worth of data, and then testing the model inference with a different day of data. What do you think of that approach? If that goes well then we could retrain the model on a month of data and see how the performance improves.

I have unzipped `cesnet-quic22.zip` into `datasets/`, resulting in a file structure as described in dataset.md. Can you write Python code to load one day (I will specify file path at the top) into Pandas? Keep in mind the files are of type `.csv.gz`. I have some more questions:
- How do I go about selecting a feature subset after that?
- What does it mean to label-encode `APP`? How do I do that?
- What does it mean to split features/labels? How do I do that?
Please implement anything related to the above questions in your code if possible, and then explain it to me.

For some context, I am running these programs on a machine with 2x 16-core AMD 7302 @ 3.00GHz (64 threads total), 128GB ECC Memory, and an NVIDIA 24GB Ampre A30 GPU. I want to maximize my use of these resources in training and running my ML model. How can this be done? As per my understanding scikit-learn doesn't even utilize the GPU.
Secondly, it seems that the data loading and matrix preparation (split the dataframe into numeric features and label-encoded targets) are repetitive steps prior to the start of training which are independent of model selection and its parameters. If that is true, can these be cached in some way so that each program run doesn't have to repeat it?

Let's first convert daily CSVs to Parquet to shrink load times and test it. Then, create a new file for training a random forest model on this data so we can benchmark it with the decision tree model. 

Earlier you mentioned "If you stay in Python/NumPy land, RAPIDS cuDF/cuML is another route (GPU DataFrame + GPU scikit-learn analogues)". I think this is a good step to take next. Keeping the existing sklearn implementations intact, in a new file, can you implement a random forest model with RAPIDS cuML? Let's train and test it with the same two days' datasets as we did for the ones prior.

Why is the sklearn random forest model working noticeably better than the cuML version? Regardless, they both aren't performing very well right now. How do I get the cuML classifier to an accuracy of >80%?

Let's first widen the feature list in `load_day.py`. Include all the features which you deem relevant to traffic classification. Make sure to cache them as well so the preprocessed data is available for use on any model we further develop. Rerun cuML forest with its existing configuration. Then, run cuML forest with deeper trees and more estimators. Compare if/how the accuracy improves.

I want to see how the cuML model improves given more training data. Let's transform the process to extract an entire week of data (W-2022-47) and train the model on it, without any sampling limits. For the testing data, we will sample a total of 1M data points from all the days of a different week (W-2022-46). Since we are dealing with such large amounts of data, please be sure to maintain the caching phase. You could do the first step of caching by having Parquet files for each of the .csv.gz files of the days. The second step of caching would be after generating feature matrices. Again, keep in mind that you have access to a 64-core CPU so make full use of it. For example when you convert the .csv.gz files to Parquet please don't limit yourself to serial executions, you can do all of them together in parallel. Similarly, during the feature engineering phase for training and evaluation data, there is a big opportunity for parallelization. You should also try to find ways to optimize code wherever you can spot redundancies.

I have reduced my parameters by such a large amount but it still seems to have no effect on reducing the memory usage. At this rate if I keep reducing parameters my model is going to get dumber and the training time will skyrocket. Why is this happening? Can you run some quick tests to estimate what the memory demands of the current parameters are?
Besides that, I'm thinking that a better approach might be to adopt some model which supports online learning so that I can incrementally update it with new data. I know that this version of decision trees or random forests will not be able to support that. What do you think we can use?

**Memory Findings**
- Ran `.python - <<'PY' ...` to read each Parquet’s `DURATION` column; totals per day match the 44 187 841-row training week.
- With 67 engineered features at float32, the feature matrix alone needs ≈ 11.0 GB on device; label vector adds ≈ 0.16 GB.
- cuML mirrors the feature matrix on GPU, so just staging data already consumes ≳70 % of a 16 GB card before training workspace is allocated.

**Why OOM Persists**
- Reducing `n_estimators`, `max_depth`, etc. shrinks model complexity but does not lower the ~11 GB dataset footprint copied to the GPU.
- cuML’s histogram-based splits allocate scratch buffers sized by `n_bins × n_features × max_batch_size × n_streams`; with your current 64×67×50×8 this is another ~7 MB per batch, replicated for gradients and class stats.
- Bootstrap sampling per tree requires additional buffers proportional to the dataset size, so even aggressive parameter cuts leave the peak close to the baseline.

**Online-Friendly Alternatives**
1. **River (Hoeffding Trees / Adaptive Random Forest)** – pure Python, supports streaming updates and concept drift handling; can reuse existing feature engineering and train in mini-batches.
2. **scikit-learn `SGDClassifier` / `PassiveAggressiveClassifier`** – CPU-based but offers `partial_fit`; you can pipeline day-by-day updates and still evaluate on cached matrices.
3. **XGBoost with `gpu_hist` + external memory** – not fully online, but it supports out-of-core training on GPUs by streaming batches from disk, keeping peak memory low.
4. **cuML Incremental Models (e.g. `SGDClassifier`)** – limited set today, but worth exploring if you need GPU acceleration and can tolerate a linear model.

If you want to keep tree ensembles, consider slicing the week into chunks, training separate models, and ensembling predictions, or investigate multi-GPU cuML via Dask.

I have a final task for you. You need to help me create `docs/midterm-report.md`, which is a progress report for this project. The document should not exceed 2,500 words. In general, the questions it should answer are: "What works and what doesn’t? What are the remaining challenges? What do you need help with?" Based on all the prompts in the provided `prompts.md` and your memory of what you have helped me with, try to reconstruct everything that we have worked on so far, step-by-step. Your report should show how we have taken incremental steps in fleshing out this project. Even if something didn't work and was replaced, talk about it briefly because it signifies learning and progress. Don't worry about formal language or format, this is more of a blueprint documenting our journey which I will later use to formally write the midterm report. Think of it as a codebase changelog with explanations of what the decisions we made were, and why we made them. For context, my next step would be to switch to an online learning model and try tweaking its parameters to get better accuracy and precision. I am yet to identify which model to use and how to implement it. My ideal plan for the project is to have a real-time demo, where I can browse the internet and have the model analyze my device's packets to classify the QUIC traffic in real time. Obviously this will involve a lot more infrastructural changes even after I've finalized and trained a model.

---

I need your help in configuring my project with conda. So far I was using pip but I switched to conda because I have to handle complex dependencies involving CUDA with Rapids cuML, and so on. One caveat is that my conda environment must be created within the /mydata/ directory, not in the default user home directory. I understand that conda has environment.yml files to create and reproduce environments. I need your help in building this sort of file because this project will be moved to new machines periodically where environments need to be rebuilt. Go through my python files, assemble a list of requirements/dependencies, and create this environment file. Keep in mind I have CUDA 12.8 installed.

Create a new Python script named `train_xgboost.py`. This script will train a GPU-accelerated XGBoost classifier on a large dataset of QUIC network traffic, stored in `datasets/cesnet-quic22`, using Dask. The script must handle datasets larger than GPU VRAM (16GB) by partitioning the data and avoiding common bottlenecks. The key requirements are:
- The current machine has 4x Nvidia Tesla V100 16GB GPUs. The script must be able to utilize all 4 GPUs in parallel by automatically detecting the number of available GPUs. However, it must also take an optional CLI argument for the number of GPUs to use.
- Use dask_cudf.read_parquet to recursively load all Parquet files from the input directory into a Dask DataFrame. See the directory structure - it consists of four weeks with seven days of data each. By default, will use the W-2022-47 week, which has Parquet files for each day's directories within it. For example, the Monday dataset is at `datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.parquet`.
- Immediately after loading, repartition the Dask DataFrame into many smaller partitions to ensure no single partition is too large for a worker's memory.
- Define and apply all feature engineering steps using Dask-native operations. Perform label encoding on the Dask DataFrame (e.g., `y = ddf['APP'].astype('category').cat.codes`) to avoid bottlenecks. You can reference `load_day.py` for examples of feature engineering and label encoding in this dataset. The dataset is also described in detail in `dataset.md`.
- Ensure the model parameters include `tree_method='hist'` and `device='cuda'`. Do not use `gpu_hist` as it is deprecated.
- After training, use the `.save_model()` method to save the classifier to the specified output path. By default, the output path should be at `datasets/cache/models`.
- After training is complete, evaluate the model on one day of data from a different week. By default, use `datasets/cesnet-quic22/W-2022-46/1_Mon/flows-20221114.parquet`.
- Reference `train_random_forest_cuml.py` to see the design patterns that should be followed. For each run, be sure to print the model parameters at the start of the run. You should also print the training time and evaluation metrics.

Now we will fix some bottlenecks to speed up our training time and use less GPU VRAM. 
1. In `main()`, we have the line `label_map = build_label_map(train_ddf)`. This function calls `.unique().compute()` which forces Dask to load and scan all input files from disk just to get a list of unique application names. Then, the script has to load and scan all input files again to do the actual training. You are effectively reading the entire dataset twice. One option could be to use Dask's built-in categorical conversion inside the `prepare_features_and_labels` function, but make sure this will be compatible with the rest of the script. Otherwise go for a different alternative.
2. You call load_dask_frame with `partition_size="256MB"`. This splits your raw, un-engineered data into 256MB chunks. You then call `prepare_features_and_labels`, which calls `engineer_features`. The `_engineer_partition` function is a partition exploder. It takes a 256MB partition and inflates it by creating dozens of new features (especially from the histograms). A 256MB raw partition can easily become 5-10GB in memory after engineering. When a GPU worker tries to process this single, massive partition, it runs out of memory. You must repartition after engineering, not before. This ensures your final partitions (the ones XGBoost actually trains on) are balanced and memory-limited.

Act as a Senior Machine Learning Engineer specializing in Network Traffic Analysis. I need you to write a complete, standalone PyTorch training pipeline for the CESNET-QUIC22 dataset. The goal is to classify encrypted QUIC traffic with >80% accuracy, improving upon a previous XGBoost baseline.
Hardware Context:
- System: Single node with 4x NVIDIA V100 GPUs (16GB VRAM), 32 CPU cores, and 128GB RAM.
- Requirement: The code must use DistributedDataParallel (DDP) to utilize all 4 GPUs efficiently.
- Requirement: Use Automatic Mixed Precision (AMP) to maximize throughput and VRAM efficiency on the V100s.
Data Specifications:
Input Format: Parquet files in `datasets/training/`. Utilize the fact that this is a column-oriented layout. The dataset fields are described in `dataset.md`.
You must write a custom Dataset class that reads these Parquet files directly. Do not rely on any existing project scripts; this must be self-contained.
Target: The label column representing the application class (105 classes).
Features to Utilize (Parse these from the schema):
- Packet Sequences (High Priority): There are columns for packet sizes, directions, and inter-packet times (IPT). These are sequences of length 30.
- Derived Stats: Flow duration, packet counts, round-trip time (RTT) estimates.
- Handshake Fields: SNI (Server Name Indication), QUIC Version, User Agent.
- Histograms: The dataset includes 2D histograms of packet sizes/directions.
Model Architecture Requirements: Since the paper suggests CNNs perform best, implement a Hybrid Multi-Modal Neural Network:
1. Sequence Branch (1D CNN):
- Input: The 3 sequences (sizes, directions, IPT) stacked as channels.
- Layers: 1D Convolutional layers with Batch Normalization and ReLU, followed by Global Max Pooling or strong dropout.
2. Static/Tabular Branch (MLP):
- Input: Flattened histograms, derived stats, and encoded handshake fields (use Embeddings for SNI/UserAgent if cardinality is high, or Hashing).
- Layers: Fully Connected layers.
3. Fusion Head: Concatenate the outputs of the Sequence Branch and Static Branch. Final classification head for the 105 classes.
Implementation Constraints:
- Data Loading: Use DistributedSampler for proper sharding across GPUs. Use num_workers to utilize the 32 CPUs for data pre-fetching.
- Robustness: Include a weighted Loss function (e.g., CrossEntropyLoss with class weights) if the dataset is imbalanced.
- Metrics: Track Accuracy and Macro-F1 score.
Structure:
CustomDataset class.
HybridCNN model class.
train_one_epoch and evaluate functions.
Main execution block handling the DDP spawning (using torch.multiprocessing).