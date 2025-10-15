# Project Proposal: Statistical Fingerprinting for QUIC Traffic Classification

## 1. Introduction and Problem Statement

Traditional network traffic classification, which relies on port numbers and protocol fields, is becoming ineffective. The widespread use of TLS encryption now bundles diverse applications like web browsing, video streaming, and file transfers onto a single port (TCP/443), making port-based identification impossible.

The recent adoption of the **QUIC protocol** by major providers like Google and Meta further complicates this issue. QUIC is an encrypted-by-default transport protocol built on UDP, and its growing use means a significant portion of internet traffic is now opaque. A network operator cannot easily distinguish between a user watching YouTube, performing a Google Search, or using Google Drive, as they all may appear as encrypted QUIC flows. This ambiguity poses challenges for network monitoring, security, and Quality of Service (QoS) management.

This project aims to address the classification of encrypted network traffic, with a specific focus on QUIC.

**Research Question:** Can we use statistical, flow-level features from packet headers to accurately classify the underlying application of an encrypted QUIC flow, without resorting to deep packet inspection?

---

## 2. Literature Review

The field has already moved beyond port-based analysis. Early work by Aceto et al. demonstrated that statistical features like **packet sizes** and **inter-arrival times** could create "fingerprints" to successfully classify mobile application traffic using machine learning.

With the rise of QUIC, research has adapted. Tong et al. applied Convolutional Neural Networks (CNNs) to QUIC classification, achieving high accuracy but at the cost of high computational overhead and low model interpretability. More recently, Luxemburk et al. showed that traditional machine learning models, such as Random Forests, can achieve performance competitive with deep learning approaches by using a feature set based on flow statistics. Their work confirms the continued relevance of the statistical fingerprinting method.

---

## 3. Proposed Approach

This project will build a machine learning pipeline to classify traffic using statistical "fingerprints." The central idea is that different applications exhibit unique traffic patterns even when encrypted. For example, video streaming typically involves a steady flow of large packets, while interactive web browsing is characterized by short bursts of smaller packets. By extracting these statistical features, a model can be trained to recognize these patterns.

Our approach consists of four main steps:
1.  **Utilize a Public Dataset:** Use a pre-labeled public dataset containing a mix of modern network traffic, including various applications running over QUIC.
2.  **Engineer Flow-Based Features:** Process packet captures to extract statistical features from traffic flows.
3.  **Train and Evaluate ML Models:** Use supervised machine learning models (e.g., Decision Trees, Random Forests) to learn the mapping between features and application labels.
4.  **Analyze QUIC Performance:** Specifically evaluate the model's ability to differentiate between various applications tunneled over QUIC.

---

## 4. Methodology and Implementation

* **Dataset:** We will use a publicly available dataset, such as the "IITH-TCN" dataset, which includes labeled QUIC traffic from applications like YouTube, Google, and Facebook.
* **Tools:** The project will be implemented in **Python**, using libraries such as `pandas` for data manipulation, `scapy` or `pyshark` for parsing PCAP files, and `scikit-learn` for ML models.

### Feature Extraction
A script will group packets into bidirectional flows. For each flow, the following features will be computed:
* **Packet Size Statistics:** Mean, median, standard deviation, min, and max packet size.
* **Inter-Arrival Time (IAT) Statistics:** Mean, median, standard deviation, min, and max IAT.
* **Flow Descriptors:** Total number of packets, total bytes, and flow duration.
* **Directional Features:** Packet and byte counts in both the forward (client-to-server) and backward (server-to-client) directions.

### Model Training
We will train at least two models:
1.  A **Decision Tree** to provide an interpretable model and analyze feature importance.
2.  A **Random Forest** or Gradient Boosting model to maximize classification accuracy.

---

## 5. Evaluation Plan

The evaluation will answer two key questions:
1.  How accurately can the model distinguish between QUIC and non-QUIC traffic?
2.  Among QUIC flows, how accurately can the model distinguish the specific application (e.g., YouTube vs. Google Search)?

We will use standard classification metrics, including **accuracy, precision, recall, and F1-score**. A confusion matrix will be used to analyze misclassifications. A key output will be a **feature importance analysis** to identify the most predictive statistical features for QUIC traffic classification.

---

## 6. Project Timeline

* **Weeks 1-2:** Dataset selection and initial research on QUIC.
* **Weeks 3-6:** Develop and debug feature extraction scripts. Train a baseline Decision Tree model and submit a midterm progress report.
* **Weeks 7-9:** Experiment with different features and models (Random Forest). Perform a detailed evaluation and generate results for the final report.
* **Week 10:** Write the final 5-6 page report and submit all source code and results.

---

## References

* Aceto, G., Ciuonzo, D., Montieri, A., and Pescape, A. "Traffic classification of mobile apps through multi-classification." In *2017 IEEE Global Communications Conference*, pages 1-6, 2017.
* Luxemburk, J., Hynek, K., and Čejka, T. "Encrypted traffic classification: the quic case." In *2023 7th Network Traffic Measurement and Analysis Conference (TMA)*, pages 1-10, 2023.
* Tong, V., Tran, H. A., Souihi, S., and Mellouk, A. "A novel quic traffic classifier based on convolutional neural networks." In *2018 IEEE Global Communications Conference*, pages 1-6, 2018.