# TreadVision

### Full Academic Title
**TreadVision: Intelligent Tyre Tread Analysis, Forensic Tread Matching and Vehicle Fitment Intelligence Using Image Processing and Machine Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-5C3EE8.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 1. Project Vision

A tyre tread pattern contains vital forensic and operational intelligence: it reveals the wear condition and safety status of a physical tyre, can identify probable tyre makes and models from impressions left at crime scenes, and connects those tyre profiles to verified vehicle fitments.

**TreadVision** is a comprehensive academic and functional computer vision system combining classical digital image processing, feature fusion, machine learning, and forensic pattern similarity matching.

> **Core Academic Tenet:**  
> TreadVision is **not** a generic black-box deep learning image classifier. The core academic value visibly stems from classical digital image processing: multi-stage preprocessing, spatial filtering, contrast-limited adaptive histogram equalization (CLAHE), Otsu and adaptive thresholding, mathematical morphology, Canny and Sobel edge orientation statistics, Gray-Level Co-occurrence Matrix (GLCM) texture analysis, contour moments, and 7 scale/rotation-invariant Hu moments. Machine learning models (SVM, Random Forest, KNN) complement and validate the extracted feature space rather than replacing it.

---

## 2. System Architecture

```text
Input Tyre / Forensic Impression Image
                 ↓
    [1] Image Validation & Normalization
                 ↓
      [2] Grayscale Transformation
                 ↓
      [3] Noise Reduction (Bilateral / Gaussian / Median)
                 ↓
   [4] Contrast Enhancement (CLAHE / HistEq)
                 ↓
 [5] Tread Segmentation (Otsu / Adaptive / Connected Components)
                 ↓
    [6] Morphological Cleanup (Opening / Closing)
                 ↓
   [7] Edge Detection (Canny / Sobel / Laplacian)
                 ↓
  ┌──────────────────────────────┴──────────────────────────────┐
  ↓                                                             ↓
[8] Feature Extraction & Fusion                [9] Forensic Keypoints (ORB)
  • 5 Edge Descriptors (Density, Orientation)    • Scale-space FAST keypoints
  • 6 GLCM Texture Features (0°, 45°, 90°, 135°) • Oriented BRIEF descriptors
  • 14 Contour & Hu Shape Moments                • Lowe's ratio test matching
  └──────────────────────────────┬──────────────────────────────┘
                                 ↓
              [10] Unified 25-D Feature Vector
                                 ↓
         ┌───────────────────────┴───────────────────────┐
         ↓                                               ↓
[Module 1: Tyre Wear Classification]          [Module 3: Forensic Tread Matcher]
  • Grouped Train/Test Stratification           • Weighted Multi-Criteria Similarity:
  • Scaled Feature Inference (SVM/RF/KNN)         S = 0.30·S_edge + 0.25·S_texture
  • Good / Moderate / Worn Classes                    + 0.25·S_shape + 0.20·S_local
         ↓                                               ↓
[Module 2: Replacement Advisor]               [Top-N Ranked Tyre Candidates]
  • Calibrated Depth Check (1.6mm legal limit)           ↓
  • Rule-Based Safety Recommendations         [Possible Vehicle Fitment Retrieval]
  • Database Size & Replacement Matching        (BMW, Audi, Mercedes, VW, Toyota)
```

---

## 3. Mathematical & Algorithmic Foundations

### 3.1 Preprocessing & Spatial Filtering
1. **Luminance Grayscale Conversion:**
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
2. **Bilateral Filter (Edge-Preserving Smoothing):**
   $$I^{\text{filtered}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) \exp\left(-\frac{\|x_i - x\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{\|I(x_i) - I(x)\|^2}{2\sigma_r^2}\right)$$
   Preserves crisp groove boundaries while attenuating high-frequency noise and asphalt/mud grain.

### 3.2 Contrast-Limited Adaptive Histogram Equalization (CLAHE)
Divides the image into contextual grids ($8 \times 8$). Local histograms are calculated, clipped at a threshold ($\beta = 2.0$) to prevent noise amplification in homogenous regions, and redistributed. Bilinear interpolation blends tile borders to eliminate boundary artifacts.

### 3.3 Tread Segmentation & Morphology
1. **Otsu's Global Bi-Level Thresholding:**
   Maximizes between-class variance:
   $$\sigma_b^2(t) = \omega_0(t)(\mu_0(t) - \mu_T)^2 + \omega_1(t)(\mu_1(t) - \mu_T)^2$$
2. **Morphological Closing:**
   $$A \bullet B = (A \oplus B) \ominus B$$
   Bridges microscopic voids in continuous circumferential grooves.

### 3.4 Edge Detection & Orientation Statistics
- **Sobel First-Order Gradients:**
  $$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$$
  Magnitude $G = \sqrt{G_x^2 + G_y^2}$, Orientation $\theta = \arctan2(G_y, G_x)$.
- **Measurable Descriptors:**
  - `total_edge_density`: ratio of edge boundary pixels to total area.
  - `horizontal_edge_density`: edges oriented circumferentially with rotation ($60^\circ \le \theta \le 120^\circ$).
  - `vertical_edge_density`: transverse lateral sipes and water displacement grooves ($\theta \le 30^\circ \lor \theta \ge 150^\circ$).
  - `edge_orientation_mean` & `edge_orientation_std`: angular dispersion across tread blocks.

### 3.5 GLCM Texture Analysis
Computes Gray-Level Co-occurrence Matrix $P(i, j | d, \theta)$ across offsets $d \in \{1, 2, 4\}$ and directions $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$:
- **Contrast:** $\sum_{i,j} |i - j|^2 P(i,j)$ (high in deep fresh tread channels, low in worn tyres)
- **Dissimilarity:** $\sum_{i,j} |i - j| P(i,j)$
- **Homogeneity:** $\sum_{i,j} \frac{P(i,j)}{1 + (i - j)^2}$ (approaches $1.0$ as tyres wear smooth)
- **Energy:** $\sum_{i,j} P(i,j)^2$
- **Correlation:** $\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$
- **ASM (Angular Second Moment):** $\text{Energy}^2$

### 3.6 Shape Descriptors & 7 Hu Invariant Moments
Moments $\mu_{pq} = \sum_{x}\sum_{y} (x - \bar{x})^p (y - \bar{y})^q f(x, y)$ normalized by $\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^\gamma}$, yielding 7 rotation, scale, and translation invariant moments ($\phi_1$ through $\phi_7$), transformed logarithmically:
$$\text{Hu}_i = -\text{sign}(\phi_i) \cdot \log_{10}(|\phi_i| + 10^{-15})$$

### 3.7 Anti-Leakage Physical Tyre Grouping
To avoid data leakage, photographs originating from the same physical tyre (`tyre_id`) are constrained strictly to either the training fold or testing fold using `GroupShuffleSplit` / `StratifiedGroupKFold`. This ensures classifiers evaluate generalization across different tyres rather than memorizing individual tyre surface defects.

### 3.8 Forensic Multi-Criteria Similarity Metric
$$\mathcal{S}_{\text{overall}} = w_{\text{edge}} \cdot \mathcal{S}_{\text{edge}} + w_{\text{texture}} \cdot \mathcal{S}_{\text{texture}} + w_{\text{shape}} \cdot \mathcal{S}_{\text{shape}} + w_{\text{local}} \cdot \mathcal{S}_{\text{local}}$$
- $w_{\text{edge}} = 0.30$: Edge density & directional orientation alignment
- $w_{\text{texture}} = 0.25$: Cosine similarity between normalized GLCM vectors
- $w_{\text{shape}} = 0.25$: Exponential Euclidean distance on Hu moments and solidity
- $w_{\text{local}} = 0.20$: Lowe's ratio test match ratio of ORB invariant keypoints

---

## 4. User-Facing Modules

### Module 1: Tyre Tread Analyzer
- Accepts photographs of actual physical tyres.
- Displays full visual inspection stages (Original, Grayscale, Denoised, Enhanced CLAHE, Segmentation Mask, Segmented Tread, Canny Edges, Sobel Gradient, Laplacian).
- Displays numerical feature tables for Edge, GLCM Texture, and Shape moments.
- Classifies wear into:
  - **Good / Low Wear**
  - **Moderately Worn**
  - **Heavily Worn**
- Displays class probability distributions and model validation metrics.

### Module 2: Tyre Replacement Advisor
- Takes analyzer wear classification and pattern metrics (Pattern Integrity %, Groove Visibility %, Edge Definition %, Texture Condition %).
- Optional manual entry for calibrated physical tread depth gauge measurements (mm).
- Benchmarks against legal and safety standards:
  - Legal limit: **1.6 mm**
  - Wet-weather advisory: **3.0 mm**
  - Winter advisory: **4.0 mm**
- Queries database to present compatible replacement tyres matching OEM dimensions (e.g., 205/55 R16) across Michelin, Continental, Bridgestone, Goodyear, Pirelli, and Hankook.

### Module 3: Tyre Tread Database & Intelligence Catalog
- Relational database (SQLite via SQLAlchemy) containing real automotive models, technical specifications, and verified vehicle fitments.
- Search and filter by manufacturer, category, tyre size, or vehicle brand.
- Extensible CSV importer with schema validation.

### Forensic Tread Matcher (Within Module 3)
- Accepts photographs of unknown tyre impressions left in **mud, soil, sand, dirt, or wet pavement**.
- Preprocesses noisy crime-scene impressions and compares against reference pattern feature indices.
- Ranks candidate tyre patterns with side-by-side visual comparisons, percentage similarity, and component score breakdowns.
- Retrieves possible/common vehicle applications (e.g. BMW 3 Series, Audi A4, Volkswagen Golf, Mercedes-Benz C-Class, Toyota Corolla).

---

## 5. Forensic Analysis Notice & Legal Disclaimers

> **Forensic Analysis Notice:**  
> TreadVision provides image-based similarity analysis for research, investigative lead generation, and educational purposes. A similarity result does not establish that a particular vehicle or tyre was present at a crime scene and must not be treated as definitive forensic evidence without professional physical examination, calibrated impression casting, and verified legal chain-of-custody procedures.

All matching outputs explicitly use non-dogmatic language:
- *"Probable tread pattern match"*
- *"Highest similarity candidate"*
- *"Potential tyre model"*
- *"Possible/common vehicle applications"*

---

## 6. Project Structure

```text
TreadVision/
│
├── app/
│   ├── main.py                     # Streamlit application entry point & router
│   ├── config.py                   # Global constants, paths, thresholds, and weights
│   │
│   ├── database/
│   │   ├── connection.py           # SQLAlchemy SQLite engine & session factory
│   │   ├── models.py               # ORM Models: Tyre, TreadPattern, TyreSize, Vehicle, Fitment, TreadFeatures
│   │   ├── seed_data.py            # Reference catalog and CV feature caching
│   │   └── importer.py             # CSV batch importer with schema validation
│   │
│   ├── image_processing/
│   │   ├── pipeline.py             # Unified 12-stage image processing pipeline
│   │   ├── preprocessing.py        # Validation, resizing, grayscale, noise reduction
│   │   ├── enhancement.py          # HistEq and CLAHE contrast enhancement
│   │   ├── segmentation.py         # Otsu and adaptive thresholding, component masking
│   │   ├── edges.py                # Canny, Sobel, Laplacian, edge densities & orientations
│   │   ├── morphology.py           # Erosion, dilation, opening, closing
│   │   ├── texture.py              # GLCM texture analysis (0°, 45°, 90°, 135°)
│   │   ├── shape.py                # Contour descriptors, solidity, 7 Hu moments
│   │   ├── feature_fusion.py       # 25-dimensional normalized feature vector
│   │   └── forensic_matcher.py     # Multi-criteria similarity engine & ORB matching
│   │
│   ├── ml/
│   │   ├── dataset.py              # Dataset scanner, anti-leakage group split
│   │   ├── models.py               # SVM, Random Forest, KNN classifier wrappers
│   │   ├── train.py                # Training pipeline, cross-validation, metrics
│   │   └── inference.py            # Feature scaling and probability prediction
│   │
│   ├── recommender/
│   │   └── replacement_advisor.py  # Rule-based replacement advisor & size matching
│   │
│   └── ui/
│       ├── theme.py                # Automotive dark modern theme & CSS
│       ├── components.py           # Feature tables, KPI cards, forensic banners
│       ├── home.py                 # Landing page with 3 major cards & telemetry
│       ├── analyzer.py             # Module 1 UI: Tyre Tread Analyzer
│       ├── replacement.py          # Module 2 UI: Tyre Replacement Advisor
│       ├── database_view.py        # Module 3 UI: Tyre Catalog & Fitments
│       └── forensic_view.py        # Module 3 Submodule: Forensic Tread Matcher
│
├── data/
│   ├── raw/
│   │   ├── tyre_wear/              # Good, moderate, worn physical tyre sets
│   │   └── forensic/
│   │       ├── impressions/        # Simulated crime-scene impressions (mud, soil, sand)
│   │       └── reference_patterns/ # High-resolution reference patterns
│   └── database/
│       └── treadvision.db          # SQLite relational database
│
├── models/                         # Serialized joblib models, scalers, and metric JSONs
├── scripts/
│   ├── generate_demo_assets.py     # Generates reference patterns, impressions & datasets
│   ├── init_db.py                  # CLI to seed database and cache CV features
│   └── train_wear_model.py         # CLI to train ML wear models with CV evaluation
│
├── tests/
│   ├── test_pipeline.py            # Unit tests for image processing & features
│   ├── test_forensic.py            # Unit tests for similarity engine & ranking
│   └── test_database.py            # Unit tests for database models & queries
│
├── pytest.ini                      # Pytest configuration
├── requirements.txt                # Project dependencies
├── run.py                          # Launcher script
└── README.md                       # Comprehensive documentation
```

---

## 7. Quickstart Guide

### 7.1 Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/TreadVision.git
cd TreadVision

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Database & Asset Initialization

Populate the SQLite database and generate calibrated reference patterns and forensic evidence samples:

```bash
python scripts/init_db.py
```

### 7.3 Train Machine Learning Models

Train Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) models on the extracted computer vision feature matrix using anti-leakage physical tyre grouping:

```bash
python scripts/train_wear_model.py
```

### 7.4 Run Automated Unit Tests

```bash
pytest
```

### 7.5 Launch TreadVision Web Application

```bash
python run.py
# or
streamlit run app/main.py
```

Open your browser at `http://localhost:8501`.

---

## 8. Experimental Evaluation

Evaluating Random Forest and Support Vector Machine classifiers on extracted 25-D classical computer vision feature vectors across calibrated validation samples yields:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | 3-Fold Stratified CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (150 trees)** | 100.0% | 1.000 | 1.000 | 1.000 | 100.0% |
| **Support Vector Classifier (RBF)** | 100.0% | 1.000 | 1.000 | 1.000 | 100.0% |
| **K-Nearest Neighbors ($k=5$)** | 100.0% | 1.000 | 1.000 | 1.000 | 96.3% |

*Notice: Current metrics reflect the project's calibrated demonstration validation set evaluated with physical-tyre group stratification (where views from the same physical tyre are strictly restricted to either the training or testing partition to avoid leakage). These results illustrate feature discriminability on the calibrated reference set and should be re-evaluated as larger operational fleet datasets are ingested.*

---

## 9. Academic Citation

If you use TreadVision in your research, computer vision projects, or forensic coursework, please cite:

```bibtex
@software{treadvision2026,
  author = {TreadVision Contributors},
  title = {TreadVision: Intelligent Tyre Tread Analysis, Forensic Tread Matching and Vehicle Fitment Intelligence Using Image Processing and Machine Learning},
  year = {2026},
  url = {https://github.com/<user>/TreadVision}
}
```
