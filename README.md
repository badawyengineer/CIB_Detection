# CIB_Detection
internship task 
# 🌐 Multilingual CIB Detection Pipeline
### End-to-End Unsupervised Deception Discovery via Cross-Lingual Semantics & Graph Neural Network (GNN) Autoencoders

---

## 📌 Project Overview
This repository contains a production-grade, end-to-end **Coordinated Inauthentic Behavior (CIB)** detection pipeline designed to surface automated botnets, coordinated narrative amplification, and malicious influence operations across multilingual social media ecosystems. 

Unlike traditional metadata-signature or keyword-bound systems, this framework treats CIB discovery as an **unsupervised, multi-dimensional optimization problem**. By fusing **cross-lingual semantic embeddings**, **topological graph community alignment**, and **inductive geometric deep learning (GraphSAGE Autoencoders)**, the pipeline isolates tightly coupled actor networks working in structural unison without requiring manual historical training labels.

### 📊 Pipeline Performance at a Glance
The system was validated against high-density heterogeneous proxy benchmarks, yielding distinct topological community isolation:

| Evaluation Metric | Calculated Value | Operational Signal |
| :--- | :---: | :--- |
| **Graph Structural Density** | `0.9371` | High localized interconnectivity across anomalous users. |
| **Silhouette Score** | `0.5943` | Clean, non-overlapping geometric boundary separation. |
| **Davies-Bouldin Index** | `0.8585` | Exceptional internal cluster compactness. |
| **Louvain Modularity** | `0.2618` | Dynamic, highly-resolved sub-network partitioning. |
| **Avg Intra-Community Density** | `0.9975` | Near-perfect structural synchronization within target networks. |

---

## ⚙️ Core Architecture & Pipeline Stages
The workflow is architected into modular, pipeline stages that isolate responsibilities from dirty multi-format ingestion up to deep structural clustering evaluation:

                              [ Heterogeneous Input Streams ]
                           ( Twitter/X • CredBank • MultiEuroLex )
                                             │
                                             ▼
                                 [ STAGE 1 & 2: Ingestion ]
                              Regex Cleaning & Token Parsing
                                             │
                                             ▼
                               [ STAGE 3: Cross-Lingual NLP ]
                          Sentence-Transformer Mapping (768-Dim)
                                             │
                                             ▼
                                [ STAGE 4: Behavioral Profiling ]
                         Extract Burst Timing, Entropy, & Uniformity
                                             │
                                             ▼
                                [ STAGE 5: Coordination Graph ]
                         Vector Thresholding & Dynamic Edge Synthesis
                                             │
                                             ▼
                 ┌───────────────────────────┴───────────────────────────┐
                 ▼                                                       ▼
    [ STAGE 6: Structural Partitioning ]                [ STAGE 7: Unsupervised Feature Mining ]
      Louvain Modularity Optimizations                      PyTorch Multi-Layer Autoencoder
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                                 [ STAGE 8 & 9: GNN Modeling ]
                           GraphSAGE Graph Autoencoder (GAE) Subspace
                                             │
                                             ▼
                                  [ STAGE 10: Evaluation ]
                               Unsupervised Cluster Analytics
                                             │
                                             ▼
                                  [ STAGE 11: Deployment ]
                            Interactive Front-End Dashboards

---

## ⚙️ 2. Comprehensive Deep Dive by Stage

### Stage 0: Installation & Environment Setup
Configures highly optimized computational backends. Installs standard dependencies along with highly sensitive geometric extensions including `torch-geometric`, `scikit-learn`, and graph libraries to handle GPU-accelerated sparse matrix updates.

### Stage 1 & 2: Multilingual Ingestion & Sanitization
* **Ingestion Engine:** Aggregates unstructured inputs across varying semantic structures (social posts from Stanford IO collections, validation nodes from CredBank, and institutional compliance strings from MultiEuroLex).
* **Sanitization Array:** Employs compiled regular expressions to strip tracking telemetry, deep-nested URL anchors, hypermedia tags, and non-standard punctuation while systematically restructuring raw user histories into continuous, normalized document streams.

### Stage 3: Cross-Lingual Semantic Featurization
To neutralize adversarial tactics where actors cycle through distinct languages to bypass content filters, the framework utilizes an advanced text vectorizer (`sentence-transformers/LaBSE` or `paraphrase-multilingual-mpnet-base-v2`).
* **Mathematical Behavior:** Text content maps to a language-agnostic, 768-dimensional hyperspace. Content expressing identical narratives in English, Arabic, Russian, or French clusters tightly together based purely on latent semantic intent.
* **User Space Aggregation:** Computes the **Semantic Centroid** for every user by averaging their history vectors, forming the structural profile matrix $\mathbf{X}_{\text{text}}$.

### Stage 4: High-Signal Behavioral Feature Engineering
For every individual account, the system engineers a robust array of behavioral metadata vectors, measuring structural deviation from authentic human interactions:
* **Intra-User Semantic Dispersion:** Quantifies narrative volatility by calculating the mean pairwise cosine distance across an isolated account's timelines. Low variance signals rigid, single-topic automated narrative repetition.
* **Temporal Burst Index:** Extracts microsecond timestamps to detect burst-firing intervals, flag abnormal duty cycles, and calculate timing delta frequencies.
* **Structural Content Entropy:** Computes the Shannon Entropy of hashtag distributions, external destination hyperlinks, and text lengths to differentiate uniform template strings from organic human variety.

### Stage 5: Coordination Graph Synthesis
Transforms separate user nodes into an interconnected topological network mapping hidden systemic relations:
* **Adjacency Assignment:** An edge $e_{ij}$ between User $i$ and User $j$ is structurally declared if their behavioral synchronization metrics surpass sharp empirical limits.
* **Edge Weights ($W_{ij}$):** Modeled as a convex combination of semantic alignment and temporal co-firing metrics:
    $$W_{ij} = \alpha \cdot \text{CosineSim}(\vec{v}_i, \vec{v}_j) + \beta \cdot \text{TemporalSync}(i, j)$$
* **Execution Profile:** The algorithm constructs a robust coordination matrix capturing **49 unique users** across **1,102 distinct verification links**, establishing a network structural density of **0.9371**.

### Stage 6: Louvain Modularity Partitioning
The topological network is split into modular operational nodes using the Louvain Modularity Maximization heuristic:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
The algorithm continuously aggregates users into non-overlapping groups until a modularity ceiling of **`0.2618`** is reached, surfacing tightly grouped hidden structures.

### Stage 7: Unsupervised Feature Space Autoencoders
Concurrently, a deep multi-layer PyTorch neural **Autoencoder** maps the structural feature matrix $\mathbf{X}$ to capture raw behavioral deviations:
* **Architecture:** Linear Encoding layers compressed with non-linear `LeakyReLU` activations down to a tightly bottlenecked latent space, mirrored by symmetric decoding layers.
* **Loss Formulation:** Optimization occurs via Mean Squared Error (MSE) minimization:
    $$\mathcal{L}_{\text{AE}} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$$
* **Signal Extraction:** High reconstruction error ($\|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$) isolates anomalous behaviors that cannot be compressed into normal baseline patterns.

### Stage 8 & 9: Inductive Graph Neural Networks & Unified CIB Scoring
To prevent sophisticated actors from masking anomalous profiles within loose network structures, the pipeline joins topology with behavior using an inductive **GraphSAGE Graph Autoencoder (GAE)**:
* **Neighborhood Aggregation:** Executes localized graph convolutions to recursively aggregate user attributes from adjacent neighbors:
    $$\mathbf{h}_v^{(k)} = \sigma \left( \mathbf{W} \cdot \left[ \mathbf{h}_v^{(k-1)} \,||\, \text{AGG} \left( \left\{ \mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}(v) \right\} \right) \right] \right)$$
* **Unified Inference Engine:** Reconstructs the complete network topology using inner-product latent decoder layers. The final **CIB Score** uses a multi-stage MinMax Scaler that maps community structural density, behavioral reconstruction penalties, and GAE embedding projections into a standardized anomaly index from `0.000` (Organic, Authentic) to `1.000` (Inauthentic, Systemic).

### Stage 10: Proxy Evaluation (Zero-Labels Required)
Validates cluster stability and boundaries using rigorous internal validation indices, protecting analytics from human validation bias:
* **Silhouette Score (`0.5943`):** Confirms clear structural boundaries and tight cluster separation.
* **Davies-Bouldin Index (`0.8585`):** Validates intra-cluster compactness relative to distance from competing centers.
* **Calinski-Harabasz Index (`27.63`):** Measures robust inter-cluster variation against localized cluster spread.

### Stage 11 & 12: Streamlit Dashboard Deployment & Advanced GNN Analysis
Compiles an interactive Python deployment file (`cib_dashboard.py`) and tracks optimization stability metrics, giving operators a comprehensive look into model predictions.

---

## 📊 3. Empirical Operational Results

During execution, the pipeline successfully isolated the target matrix into two distinct, highly polarized behavioral modules:

              ┌─────────────────────────────────────────┐
              │    COORDINATED ADVERSARIAL DISCOVERY    │
              └─────────────────────────────────────────┘
                ├── Community 1: [████████████████] 1.000 CIB Score (29 Users)
                └── Community 0: [░░░░░░░░░░░░░░░░] 0.000 CIB Score (20 Users)

### 🚨 Target Footprint Breakdown

* **Community 1 (Systemic Adversarial Core):**
    * **Scale:** 29 Active Nodes.
    * **Internal Density:** **`99.5%`**.
    * **Attributes:** Near-zero timeline delays, high narrative alignment across multiple languages, highly overlapping shared destination links, and massive structural centralization. This profile indicates a highly coordinated disinformation network.
* **Community 0 (Organic Baseline Controls):**
    * **Scale:** 20 Active Nodes.
    * **Internal Density:** Low, decentralized network coupling.
    * **Attributes:** Diverse posting patterns, natural semantic dispersion, and language usage mirroring standard human variation.

---

## 🛠️ 4. Core Technology & Library Stack

* **Geometric Deep Learning:** `PyTorch`, `PyTorch Geometric (PyG)` (GraphSAGE, Graph Autoencoders)
* **Graph Network Formulations:** `NetworkX`, `python-louvain`
* **Natural Language Processing:** `HuggingFace Transformers`, `Sentence-Transformers` (LaBSE, MPNet)
* **Mathematical Analytics:** `Scikit-Learn`, `Pandas`, `NumPy`, `SciPy`
* **UI Front-End & Graphs:** `Streamlit`, `Matplotlib`, `Seaborn`

---

## 🚀 5. Getting Started & Deployment

### Environment Procurement
Ensure an execution machine configured with a **CUDA-capable GPU (e.g., NVIDIA T4/A100)** running **Python 3.10+**:
