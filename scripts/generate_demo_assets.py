"""
TreadVision: Synthetic Asset and Demo Dataset Generator
Generates realistic computer vision reference patterns, forensic crime scene impressions,
and calibrated physical-tyre-grouped wear datasets.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    REFERENCE_PATTERNS_DIR,
    FORENSIC_IMPRESSIONS_DIR,
    WEAR_DATA_DIR
)


def create_base_tread_canvas(h=512, w=512, bg_intensity=135):
    """Create physical matte rubber tread surface with micro-roughness."""
    base = np.full((h, w), bg_intensity, dtype=np.uint8)
    noise = np.random.normal(0, 5, (h, w)).astype(np.int16)
    textured = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return textured


def render_primacy4_pattern(wear_level: float = 0.0) -> np.ndarray:
    """
    Renders Michelin Primacy 4 style asymmetric tread:
    4 longitudinal grooves, lateral curved sipes, outside shoulder blocks.
    wear_level: 0.0 (fresh/new) to 1.0 (completely bald/worn).
    """
    bg_intensity = int(138 - wear_level * 10)
    img = create_base_tread_canvas(512, 512, bg_intensity=bg_intensity)
    groove_depth = int(20 + wear_level * 36)
    groove_positions = [110, 200, 310, 410]
    groove_width = max(8, int(18 * (1.0 - wear_level * 0.40)))

    for x in groove_positions:
        cv2.line(img, (x, 0), (x, 512), color=groove_depth, thickness=groove_width)

    # Lateral angled sipes on outer ribs (continuously degraded with wear)
    sipe_alpha = max(0.0, float(1.0 - (wear_level / 0.78)**1.3))
    if sipe_alpha > 0.01:
        sipe_canvas = img.copy()
        sipe_depth = int(28 + wear_level * 45)
        sipe_spacing = 32
        thick = 2 if wear_level < 0.28 else 1
        rib_factors = [0.85, 1.10, 1.25, 1.10, 0.85]
        for y in range(0, 512, sipe_spacing):
            cv2.line(sipe_canvas, (10, y), (100, y + 15), color=int(sipe_depth + (1.0 - rib_factors[0]) * 15), thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (120, y + 10), (190, y + 25), color=int(sipe_depth + (1.0 - rib_factors[1]) * 15), thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (210, y + 5), (300, y + 20), color=int(sipe_depth + (1.0 - rib_factors[2]) * 15), thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (320, y + 15), (400, y + 5), color=int(sipe_depth + (1.0 - rib_factors[3]) * 15), thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (420, y + 10), (500, y + 25), color=int(sipe_depth + (1.0 - rib_factors[4]) * 15), thickness=thick, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(sipe_canvas, sipe_alpha, img, 1.0 - sipe_alpha, 0)

    noise_sigma = max(1.8, 4.5 * (1.0 - wear_level * 0.60))
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.35)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def render_premiumcontact6_pattern(wear_level: float = 0.0) -> np.ndarray:
    """
    Renders Continental PremiumContact 6 style asymmetric tread:
    3 wide longitudinal channels, macro-block shoulder, asymmetric chevron ribs.
    """
    bg_intensity = int(138 - wear_level * 10)
    img = create_base_tread_canvas(512, 512, bg_intensity=bg_intensity)
    groove_depth = int(20 + wear_level * 36)
    groove_positions = [140, 260, 380]
    groove_width = max(10, int(24 * (1.0 - wear_level * 0.40)))

    for x in groove_positions:
        cv2.line(img, (x, 0), (x, 512), color=groove_depth, thickness=groove_width)

    sipe_alpha = max(0.0, float(1.0 - (wear_level / 0.78)**1.3))
    if sipe_alpha > 0.01:
        sipe_canvas = img.copy()
        sipe_depth = int(28 + wear_level * 45)
        sipe_spacing = 40
        thick = 2 if wear_level < 0.28 else 1
        for y in range(0, 512, sipe_spacing):
            cv2.line(sipe_canvas, (10, y), (130, y + 20), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            pts = np.array([[155, y + 10], [200, y + 25], [245, y + 10]], np.int32)
            cv2.polylines(sipe_canvas, [pts], isClosed=False, color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (275, y + 20), (365, y + 5), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (395, y + 5), (500, y + 30), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(sipe_canvas, sipe_alpha, img, 1.0 - sipe_alpha, 0)

    noise_sigma = max(1.8, 4.5 * (1.0 - wear_level * 0.60))
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.35)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def render_turanza_pattern(wear_level: float = 0.0) -> np.ndarray:
    """
    Renders Bridgestone Turanza T005 style tread:
    4 ribs, high-angle shoulder siping, continuous center rib.
    """
    bg_intensity = int(138 - wear_level * 10)
    img = create_base_tread_canvas(512, 512, bg_intensity=bg_intensity)
    groove_depth = int(20 + wear_level * 36)
    groove_positions = [120, 215, 305, 400]
    groove_width = max(8, int(16 * (1.0 - wear_level * 0.40)))

    for x in groove_positions:
        cv2.line(img, (x, 0), (x, 512), color=groove_depth, thickness=groove_width)

    sipe_alpha = max(0.0, float(1.0 - (wear_level / 0.78)**1.3))
    if sipe_alpha > 0.01:
        sipe_canvas = img.copy()
        sipe_depth = int(28 + wear_level * 45)
        sipe_spacing = 28
        thick = 2 if wear_level < 0.28 else 1
        for y in range(0, 512, sipe_spacing):
            cv2.line(sipe_canvas, (15, y), (110, y + 30), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (130, y + 15), (205, y), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (315, y), (390, y + 20), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (410, y + 25), (495, y), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(sipe_canvas, sipe_alpha, img, 1.0 - sipe_alpha, 0)

    noise_sigma = max(1.8, 4.5 * (1.0 - wear_level * 0.60))
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.35)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def render_efficientgrip_pattern(wear_level: float = 0.0) -> np.ndarray:
    """
    Renders Goodyear EfficientGrip Performance 2 style tread:
    Curved lateral hydro-channels and 4 circumferential ribs.
    """
    bg_intensity = int(138 - wear_level * 10)
    img = create_base_tread_canvas(512, 512, bg_intensity=bg_intensity)
    groove_depth = int(20 + wear_level * 36)
    groove_positions = [105, 205, 315, 415]
    groove_width = max(8, int(17 * (1.0 - wear_level * 0.40)))

    for x in groove_positions:
        cv2.line(img, (x, 0), (x, 512), color=groove_depth, thickness=groove_width)

    sipe_alpha = max(0.0, float(1.0 - (wear_level / 0.78)**1.3))
    if sipe_alpha > 0.01:
        sipe_canvas = img.copy()
        sipe_depth = int(28 + wear_level * 45)
        sipe_spacing = 34
        thick = 2 if wear_level < 0.28 else 1
        for y in range(0, 512, sipe_spacing):
            cv2.ellipse(sipe_canvas, (50, y), (50, 20), 0, 0, 90, sipe_depth, thick, lineType=cv2.LINE_AA)
            cv2.ellipse(sipe_canvas, (460, y), (50, 20), 0, 90, 180, sipe_depth, thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (120, y + 10), (195, y + 25), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (220, y + 20), (305, y + 5), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(sipe_canvas, sipe_alpha, img, 1.0 - sipe_alpha, 0)

    noise_sigma = max(1.8, 4.5 * (1.0 - wear_level * 0.60))
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.35)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def render_cinturato_pattern(wear_level: float = 0.0) -> np.ndarray:
    """
    Renders Pirelli Cinturato P7 style tread:
    4 wide longitudinal grooves with compact center blocks.
    """
    bg_intensity = int(138 - wear_level * 10)
    img = create_base_tread_canvas(512, 512, bg_intensity=bg_intensity)
    groove_depth = int(20 + wear_level * 36)
    groove_positions = [115, 210, 310, 405]
    groove_width = max(8, int(19 * (1.0 - wear_level * 0.40)))

    for x in groove_positions:
        cv2.line(img, (x, 0), (x, 512), color=groove_depth, thickness=groove_width)

    sipe_alpha = max(0.0, float(1.0 - (wear_level / 0.78)**1.3))
    if sipe_alpha > 0.01:
        sipe_canvas = img.copy()
        sipe_depth = int(28 + wear_level * 45)
        sipe_spacing = 30
        thick = 2 if wear_level < 0.28 else 1
        for y in range(0, 512, sipe_spacing):
            cv2.line(sipe_canvas, (10, y + 10), (105, y + 20), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.rectangle(sipe_canvas, (130, y + 5), (195, y + 25), color=sipe_depth, thickness=thick)
            cv2.rectangle(sipe_canvas, (225, y + 10), (295, y + 25), color=sipe_depth, thickness=thick)
            cv2.line(sipe_canvas, (325, y + 20), (395, y + 10), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            cv2.line(sipe_canvas, (420, y + 15), (500, y + 5), color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(sipe_canvas, sipe_alpha, img, 1.0 - sipe_alpha, 0)

    noise_sigma = max(1.8, 4.5 * (1.0 - wear_level * 0.60))
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.35)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def render_ventus_pattern(wear_level: float = 0.0) -> np.ndarray:
    """
    Renders Hankook Ventus Prime 4 style tread:
    Chamfered groove edges, 3D zigzag lateral siping.
    """
    bg_intensity = int(138 - wear_level * 10)
    img = create_base_tread_canvas(512, 512, bg_intensity=bg_intensity)
    groove_depth = int(20 + wear_level * 36)
    groove_positions = [110, 205, 305, 410]
    groove_width = max(8, int(18 * (1.0 - wear_level * 0.40)))

    for x in groove_positions:
        cv2.line(img, (x, 0), (x, 512), color=groove_depth, thickness=groove_width)

    sipe_alpha = max(0.0, float(1.0 - (wear_level / 0.78)**1.3))
    if sipe_alpha > 0.01:
        sipe_canvas = img.copy()
        sipe_depth = int(28 + wear_level * 45)
        sipe_spacing = 36
        thick = 2 if wear_level < 0.28 else 1
        for y in range(0, 512, sipe_spacing):
            pts1 = np.array([[15, y], [55, y+15], [95, y]], np.int32)
            cv2.polylines(sipe_canvas, [pts1], False, color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            pts2 = np.array([[125, y+5], [160, y+20], [195, y+5]], np.int32)
            cv2.polylines(sipe_canvas, [pts2], False, color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            pts3 = np.array([[220, y+10], [260, y+25], [295, y+10]], np.int32)
            cv2.polylines(sipe_canvas, [pts3], False, color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            pts4 = np.array([[320, y+5], [360, y+20], [395, y+5]], np.int32)
            cv2.polylines(sipe_canvas, [pts4], False, color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
            pts5 = np.array([[420, y], [460, y+15], [500, y]], np.int32)
            cv2.polylines(sipe_canvas, [pts5], False, color=sipe_depth, thickness=thick, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(sipe_canvas, sipe_alpha, img, 1.0 - sipe_alpha, 0)

    noise_sigma = max(1.8, 4.5 * (1.0 - wear_level * 0.60))
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.35)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_forensic_impression(
    base_bgr: np.ndarray,
    medium: str = "mud",
    illumination_gradient: bool = True,
    rotation_deg: float = 0.0,
    crop_factor: float = 0.95
) -> np.ndarray:
    """
    Simulates realistic environmental impression in mud, sandy soil, or gravel:
    - adds medium color tint (earthy brown/clay)
    - adds irregular soil particulate noise
    - adds soft gradient illumination (shadows)
    - slight rotation and perspective warp
    """
    h, w = base_bgr.shape[:2]
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)

    # Invert tread contact: tyre impression depresses into soil, where ribs leave grooves
    inverted = cv2.bitwise_not(gray)

    # Environmental medium tinting
    if medium == "mud":
        # Deep wet mud: rich dark brown-umber
        tint_bgr = np.array([35, 55, 80], dtype=np.uint8)
    elif medium == "soil":
        # Dry sandy soil: light ochre-tan
        tint_bgr = np.array([55, 85, 115], dtype=np.uint8)
    elif medium == "sand":
        # Coarse beach/dune sand: pale yellow-tan
        tint_bgr = np.array([80, 115, 140], dtype=np.uint8)
    else:
        # Wet asphalt/dirt
        tint_bgr = np.array([45, 60, 75], dtype=np.uint8)

    # Composite textured surface
    textured_soil = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        channel = (inverted.astype(np.float32) / 255.0) * tint_bgr[c] * 1.6
        textured_soil[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Add realistic environmental soil grit / granular noise
    soil_noise = np.random.normal(0, 14, (h, w, 3)).astype(np.int16)
    impr = np.clip(textured_soil.astype(np.int16) + soil_noise, 0, 255).astype(np.uint8)

    # Irregular lighting gradient (sunlight / flashlight / shadows)
    if illumination_gradient:
        X, Y = np.meshgrid(np.linspace(0.7, 1.2, w), np.linspace(0.8, 1.1, h))
        gradient = (X * Y)[:, :, np.newaxis]
        impr = np.clip(impr.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    # Slight rotation
    if abs(rotation_deg) > 0.1:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_deg, 1.0)
        impr = cv2.warpAffine(impr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return impr


def generate_all_demo_assets():
    """
    Generate all reference patterns, forensic impressions, and wear datasets.
    """
    REFERENCE_PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
    FORENSIC_IMPRESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (WEAR_DATA_DIR / "good").mkdir(parents=True, exist_ok=True)
    (WEAR_DATA_DIR / "moderate").mkdir(parents=True, exist_ok=True)
    (WEAR_DATA_DIR / "worn").mkdir(parents=True, exist_ok=True)

    print("Generating reference tread patterns...")
    pattern_generators = {
        "michelin_primacy4.jpg": render_primacy4_pattern,
        "continental_premiumcontact6.jpg": render_premiumcontact6_pattern,
        "bridgestone_turanza_t005.jpg": render_turanza_pattern,
        "goodyear_efficientgrip2.jpg": render_efficientgrip_pattern,
        "pirelli_cinturato_p7.jpg": render_cinturato_pattern,
        "hankook_ventus_prime4.jpg": render_ventus_pattern,
    }

    ref_images = {}
    for filename, gen_fn in pattern_generators.items():
        img = gen_fn(wear_level=0.0)
        save_path = REFERENCE_PATTERNS_DIR / filename
        cv2.imwrite(str(save_path), img)
        ref_images[filename] = img
        print(f"  [+] Reference: {save_path.name}")

    print("\nGenerating simulated crime-scene impressions...")
    crime_scenes = [
        ("impression_crime_scene_01_mud_michelin.jpg", ref_images["michelin_primacy4.jpg"], "mud", -2.5),
        ("impression_crime_scene_02_soil_continental.jpg", ref_images["continental_premiumcontact6.jpg"], "soil", 3.0),
        ("impression_crime_scene_03_dirt_bridgestone.jpg", ref_images["bridgestone_turanza_t005.jpg"], "dirt", 1.5),
        ("impression_crime_scene_04_sand_goodyear.jpg", ref_images["goodyear_efficientgrip2.jpg"], "sand", -4.0),
        ("impression_crime_scene_05_mud_pirelli.jpg", ref_images["pirelli_cinturato_p7.jpg"], "mud", 2.0),
    ]

    for filename, base_img, medium, rot in crime_scenes:
        impr = make_forensic_impression(base_img, medium=medium, rotation_deg=rot)
        save_path = FORENSIC_IMPRESSIONS_DIR / filename
        cv2.imwrite(str(save_path), impr)
        print(f"  [+] Forensic Impression: {save_path.name}")

    print("\nGenerating calibrated physical-tyre grouped wear dataset...")
    # Generate images grouped by physical tyre id to ensure anti-leakage grouping!
    # Physical tyres:
    # Good: tyre_g01, tyre_g02, tyre_g03, tyre_g04 (wear 0.0 - 0.20)
    # Moderate: tyre_m01, tyre_m02, tyre_m03, tyre_m04 (wear 0.40 - 0.60)
    # Worn: tyre_w01, tyre_w02, tyre_w03, tyre_w04 (wear 0.80 - 0.95)

    tyre_types = [
        render_primacy4_pattern,
        render_premiumcontact6_pattern,
        render_turanza_pattern,
        render_efficientgrip_pattern
    ]

    # Generate Good class (wear 0.08 - 0.20: near-new tread, intact sipes & deep channels)
    for i in range(1, 5):
        tyre_id = f"tyre_g0{i}"
        gen = tyre_types[(i - 1) % len(tyre_types)]
        for angle_idx, rot in enumerate([0.0, 3.5, -3.5]):
            wear = np.random.uniform(0.08, 0.20)
            base = gen(wear_level=wear)
            if abs(rot) > 0.1:
                M = cv2.getRotationMatrix2D((256, 256), rot, 1.0)
                base = cv2.warpAffine(base, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
            # Realistic sensor noise and lighting variation (standardized across classes)
            gain = np.random.uniform(0.95, 1.05)
            noise = np.random.normal(0, np.random.uniform(2.0, 4.0), base.shape).astype(np.float32)
            final_img = np.clip(base.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
            fname = f"{tyre_id}_view{angle_idx + 1}.jpg"
            cv2.imwrite(str(WEAR_DATA_DIR / "good" / fname), final_img)

    # Generate Moderate class (wear 0.42 - 0.56: mid-life tread, thinned sipes, degraded relief)
    for i in range(1, 5):
        tyre_id = f"tyre_m0{i}"
        gen = tyre_types[(i - 1) % len(tyre_types)]
        for angle_idx, rot in enumerate([0.0, 3.5, -3.5]):
            wear = np.random.uniform(0.42, 0.56)
            base = gen(wear_level=wear)
            if abs(rot) > 0.1:
                M = cv2.getRotationMatrix2D((256, 256), rot, 1.0)
                base = cv2.warpAffine(base, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
            gain = np.random.uniform(0.95, 1.05)
            noise = np.random.normal(0, np.random.uniform(2.0, 4.0), base.shape).astype(np.float32)
            final_img = np.clip(base.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
            fname = f"{tyre_id}_view{angle_idx + 1}.jpg"
            cv2.imwrite(str(WEAR_DATA_DIR / "moderate" / fname), final_img)

    # Generate Worn class (wear 0.80 - 0.92: legal-limit/worn tread, vanished sipes, shallow grooves)
    for i in range(1, 5):
        tyre_id = f"tyre_w0{i}"
        gen = tyre_types[(i - 1) % len(tyre_types)]
        for angle_idx, rot in enumerate([0.0, 3.5, -3.5]):
            wear = np.random.uniform(0.80, 0.92)
            base = gen(wear_level=wear)
            if abs(rot) > 0.1:
                M = cv2.getRotationMatrix2D((256, 256), rot, 1.0)
                base = cv2.warpAffine(base, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
            gain = np.random.uniform(0.95, 1.05)
            noise = np.random.normal(0, np.random.uniform(2.0, 4.0), base.shape).astype(np.float32)
            final_img = np.clip(base.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
            fname = f"{tyre_id}_view{angle_idx + 1}.jpg"
            cv2.imwrite(str(WEAR_DATA_DIR / "worn" / fname), final_img)

    print("  [+] Wear dataset generated successfully with physical tyre grouping.")
    print("Done generating all demo assets.")


if __name__ == "__main__":
    generate_all_demo_assets()
