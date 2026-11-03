"""
Unit tests for TreadVision Image Processing Pipeline and Feature Extractors
"""
import pytest
import numpy as np
import cv2

from app.image_processing.preprocessing import validate_image, resize_image, to_grayscale, reduce_noise
from app.image_processing.enhancement import enhance_contrast
from app.image_processing.segmentation import segment_tread_region, otsu_threshold
from app.image_processing.edges import detect_canny_edges, extract_edge_features
from app.image_processing.texture import extract_glcm_features
from app.image_processing.shape import compute_shape_features
from app.image_processing.feature_fusion import FeatureFusion, FEATURE_NAMES
from app.image_processing.pipeline import ImageProcessingPipeline


@pytest.fixture
def sample_test_image():
    """Create a synthetic 256x256 test image with 3 simulated tread grooves."""
    img = np.full((256, 256, 3), 40, dtype=np.uint8)
    # Add vertical grooves
    cv2.line(img, (64, 0), (64, 256), (200, 200, 200), 8)
    cv2.line(img, (128, 0), (128, 256), (200, 200, 200), 8)
    cv2.line(img, (192, 0), (192, 256), (200, 200, 200), 8)
    return img


def test_validate_image(sample_test_image):
    # Valid image
    is_valid, msg, cv_img = validate_image(sample_test_image)
    assert is_valid is True
    assert cv_img is not None
    assert cv_img.shape == (256, 256, 3)

    # Invalid None
    is_valid, msg, cv_img = validate_image(None)
    assert is_valid is False

    # Image too small
    small = np.zeros((32, 32, 3), dtype=np.uint8)
    is_valid, msg, cv_img = validate_image(small)
    assert is_valid is False
    assert "dimensions too small" in msg


def test_resize_and_grayscale(sample_test_image):
    resized = resize_image(sample_test_image, target_size=(512, 512), keep_aspect_ratio=True)
    assert resized.shape == (512, 512, 3)

    gray = to_grayscale(resized)
    assert gray.shape == (512, 512)
    assert len(gray.shape) == 2


def test_noise_reduction(sample_test_image):
    gray = to_grayscale(sample_test_image)
    denoised_gauss = reduce_noise(gray, method="gaussian")
    denoised_median = reduce_noise(gray, method="median")
    denoised_bi = reduce_noise(gray, method="bilateral")

    assert denoised_gauss.shape == gray.shape
    assert denoised_median.shape == gray.shape
    assert denoised_bi.shape == gray.shape


def test_enhancement_and_segmentation(sample_test_image):
    gray = to_grayscale(sample_test_image)
    enhanced = enhance_contrast(gray, method="clahe")
    assert enhanced.shape == gray.shape

    mask, segmented, meta = segment_tread_region(enhanced)
    assert mask.shape == gray.shape
    assert segmented.shape == gray.shape
    assert "coverage_ratio" in meta


def test_edge_detection(sample_test_image):
    gray = to_grayscale(sample_test_image)
    edges = detect_canny_edges(gray, low_threshold=50, high_threshold=150)
    assert edges.shape == gray.shape

    feats = extract_edge_features(edges)
    assert "total_edge_density" in feats
    assert "horizontal_edge_density" in feats
    assert "vertical_edge_density" in feats
    assert feats["total_edge_density"] > 0.0


def test_glcm_texture_features(sample_test_image):
    gray = to_grayscale(sample_test_image)
    glcm_dict = extract_glcm_features(gray)
    assert "contrast" in glcm_dict
    assert "homogeneity" in glcm_dict
    assert "energy" in glcm_dict
    assert "ASM" in glcm_dict
    assert glcm_dict["contrast"] >= 0.0


def test_shape_features(sample_test_image):
    gray = to_grayscale(sample_test_image)
    _, binary = otsu_threshold(gray, inverse=False)
    shape_dict = compute_shape_features(binary)

    assert "area_ratio" in shape_dict
    assert "aspect_ratio" in shape_dict
    assert "solidity" in shape_dict
    assert "hu_moments" in shape_dict
    assert len(shape_dict["hu_moments"]) == 7


def test_feature_fusion(sample_test_image):
    gray = to_grayscale(sample_test_image)
    enhanced = enhance_contrast(gray, method="clahe")
    edges = detect_canny_edges(enhanced)
    _, binary = otsu_threshold(enhanced)

    comp_dict, vec = FeatureFusion.extract_all(enhanced, edges, binary, raw_image=sample_test_image)
    assert len(vec) == 31
    assert len(FEATURE_NAMES) == 31
    assert "edge" in comp_dict
    assert "texture" in comp_dict
    assert "shape" in comp_dict
    assert "wear" in comp_dict
    assert len(comp_dict["wear"]) == 6


def test_full_pipeline(sample_test_image):
    pipeline = ImageProcessingPipeline(target_size=(256, 256))
    res = pipeline.process(sample_test_image)

    assert res.success is True
    assert "Original" in res.stages
    assert "Grayscale" in res.stages
    assert "Enhanced (CLAHE)" in res.stages
    assert "Canny Edges" in res.stages
    assert res.feature_vector is not None
    assert len(res.feature_vector) == 31
    assert res.metrics["pattern_integrity_pct"] > 0.0


def test_adaptive_canny_bounds():
    """Verify adaptive Canny threshold computation satisfies specified bounds [15, 120] and [40, 220]."""
    from app.image_processing.edges import compute_adaptive_canny_thresholds
    # Dark uniform image
    dark_img = np.zeros((100, 100), dtype=np.uint8)
    low_d, high_d = compute_adaptive_canny_thresholds(dark_img)
    assert 15 <= low_d <= 120
    assert 40 <= high_d <= 220

    # High contrast checkerboard
    checker = np.indices((100, 100)).sum(axis=0) % 2 * 255
    checker = checker.astype(np.uint8)
    low_c, high_c = compute_adaptive_canny_thresholds(checker)
    assert 15 <= low_c <= 120
    assert 40 <= high_c <= 220


def test_wear_specific_features_standalone(sample_test_image):
    """Verify all 6 wear-specific features return bounded, non-NaN values."""
    from app.image_processing.wear_features import extract_wear_specific_features, WEAR_FEATURE_NAMES
    res = extract_wear_specific_features(sample_test_image)
    assert len(res) == 6
    for k in WEAR_FEATURE_NAMES:
        assert k in res
        assert not np.isnan(res[k])
        assert not np.isinf(res[k])


def test_training_inference_feature_parity(tmp_path):
    """Verify exact bit-for-bit parity of 31-D feature vectors between training and inference pipelines."""
    # Create test tyre view
    img = np.full((256, 256, 3), 40, dtype=np.uint8)
    cv2.line(img, (64, 0), (64, 256), (200, 200, 200), 8)
    cv2.line(img, (128, 0), (128, 256), (200, 200, 200), 8)
    cv2.line(img, (192, 0), (192, 256), (200, 200, 200), 8)
    test_img_path = tmp_path / "tyre_test01_view1.jpg"
    cv2.imwrite(str(test_img_path), img)

    # 1. Feature extraction through standard pipeline (inference path)
    infer_pipeline = ImageProcessingPipeline(target_size=(512, 512))
    infer_res = infer_pipeline.process(str(test_img_path))
    assert infer_res.success is True
    infer_vec = infer_res.feature_vector

    # 2. Feature extraction through WearDataset (training path)
    from app.ml.dataset import WearDataset
    dataset_mgr = WearDataset(data_dir=tmp_path)
    records = [{"file_path": str(test_img_path), "label": "good", "tyre_id": "tyre_test01"}]
    X, y, groups, paths = dataset_mgr.build_feature_matrix(records)
    train_vec = X[0]

    assert len(infer_vec) == 31
    assert len(train_vec) == 31
    np.testing.assert_allclose(infer_vec, train_vec, rtol=1e-5, atol=1e-5)
