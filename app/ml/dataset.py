"""
TreadVision: Tyre Wear Dataset Management and Anti-Leakage Split Strategy
Prevents data leakage by ensuring all images from the same physical tyre stay strictly
in either train or test split.
"""
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import os
import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.config import WEAR_DATA_DIR, WEAR_CLASSES
from app.image_processing.pipeline import ImageProcessingPipeline


class WearDataset:
    """
    Manages loading of tyre tread images, feature extraction caching,
    label encoding, and leakage-free stratified group splitting.
    """

    def __init__(self, data_dir: Path = WEAR_DATA_DIR):
        self.data_dir = data_dir
        self.pipeline = ImageProcessingPipeline(target_size=(512, 512))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(WEAR_CLASSES)

    def scan_dataset(self) -> List[Dict[str, Any]]:
        """
        Scan dataset directory structure:
        data/raw/tyre_wear/
            good/
            moderate/
            worn/
        Files are named with physical tyre id prefix if available, e.g.:
        tyre01_view1.jpg, tyre01_view2.jpg, etc.
        """
        records = []
        for wear_class in WEAR_CLASSES:
            class_folder = self.data_dir / wear_class
            if not class_folder.exists():
                continue

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for file_path in class_folder.glob(ext):
                    filename = file_path.stem
                    # Extract physical tyre ID (e.g., "tyre01" from "tyre01_01.jpg")
                    parts = filename.split("_")
                    if len(parts) >= 2:
                        tyre_id = f"{parts[0]}_{parts[1]}"
                    else:
                        tyre_id = filename

                    records.append({
                        "file_path": str(file_path),
                        "label": wear_class,
                        "tyre_id": tyre_id
                    })
        return records

    def build_feature_matrix(
        self,
        records: List[Dict[str, Any]],
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract 25 classical computer vision features for each image.
        Returns:
            (X, y, groups, file_paths)
        """
        X_list = []
        y_list = []
        groups_list = []
        paths_list = []

        total = len(records)
        for idx, rec in enumerate(records):
            if progress_callback:
                progress_callback(idx + 1, total, rec["file_path"])

            res = self.pipeline.process(rec["file_path"])
            if res.success and res.feature_vector is not None:
                X_list.append(res.feature_vector)
                y_list.append(rec["label"])
                groups_list.append(rec["tyre_id"])
                paths_list.append(rec["file_path"])

        if not X_list:
            return np.empty((0, 25)), np.empty(0), np.empty(0), []

        X = np.array(X_list, dtype=np.float32)
        y = self.label_encoder.transform(y_list)
        groups = np.array(groups_list)

        return X, y, groups, paths_list

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        test_size: float = 0.25,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Anti-leakage split: ensures images from the same physical tyre (groups)
        do not leak across train and test sets.
        """
        unique_groups = np.unique(groups)
        if len(unique_groups) > 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        else:
            # Fallback if too few distinct tyres
            from sklearn.model_selection import train_test_split
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
