"""
TreadVision: Configuration and Path Constants
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = DATA_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "treadvision.db"

# Dataset Paths
RAW_DATA_DIR = DATA_DIR / "raw"
WEAR_DATA_DIR = RAW_DATA_DIR / "tyre_wear"
FORENSIC_IMPRESSIONS_DIR = RAW_DATA_DIR / "forensic" / "impressions"
REFERENCE_PATTERNS_DIR = RAW_DATA_DIR / "forensic" / "reference_patterns"
EVIDENCE_UPLOADS_DIR = RAW_DATA_DIR / "forensic" / "evidence_uploads"
EVIDENCE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Image Processing Defaults
DEFAULT_IMAGE_SIZE = (512, 512)
CANNY_LOW_THRESHOLD = 35
CANNY_HIGH_THRESHOLD = 90
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# GLCM Texture Settings
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, 0.785398, 1.570796, 2.356194]  # 0, 45, 90, 135 degrees in radians
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# Forensic Matching Weights (sum to 1.0)
FORENSIC_WEIGHTS = {
    "edge": 0.30,
    "texture": 0.25,
    "shape": 0.25,
    "local": 0.20
}

# Tyre Wear Classes
WEAR_CLASSES = ["good", "moderate", "worn"]
WEAR_LABELS = {
    "good": "Good / Low Wear",
    "moderate": "Moderately Worn",
    "worn": "Heavily Worn"
}

# Legal & Recommended Tread Depth Thresholds (in mm)
LEGAL_MINIMUM_DEPTH_MM = 1.6
WET_WEATHER_RECOMMENDED_DEPTH_MM = 3.0
WINTER_RECOMMENDED_DEPTH_MM = 4.0
NEW_TYRE_TYPICAL_DEPTH_MM = 8.0

# Database Connection URI
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
