"""Tests du prétraitement des régions et de l'extraction des descripteurs."""
import numpy as np
import pytest

from src.features.extractors import ALL_BLOCKS, FeatureConfig, extract_feature_blocks, extract_features
from src.features.preprocessing import PreprocessConfig, crop_region, preprocess_region, region_patch

RNG = np.random.default_rng(0)
IMAGE = RNG.integers(0, 256, (200, 300, 3), dtype=np.uint8)


def test_crop_without_margin_is_exact_bbox():
    crop = crop_region(IMAGE, (10, 20, 50, 30), margin=0.0)
    assert crop.shape == (30, 50, 3)
    assert np.array_equal(crop, IMAGE[20:50, 10:60])


def test_crop_with_margin_is_clipped_to_image():
    crop = crop_region(IMAGE, (0, 0, 100, 50), margin=0.5)     # marge de 50 px qui sort à gauche/en haut
    assert crop.shape == (100, 150, 3)
    crop = crop_region(IMAGE, (295, 195, 10, 10), margin=0.0)  # bbox qui dépasse en bas à droite
    assert crop.shape == (5, 5, 3)


@pytest.mark.parametrize("mode", ["stretch", "pad"])
@pytest.mark.parametrize("shape", [(30, 200, 3), (300, 12, 3), (1, 1, 3), (500, 400, 3)])
def test_preprocess_always_returns_square_uint8_patch(mode, shape):
    region = RNG.integers(0, 256, shape, dtype=np.uint8)
    patch = preprocess_region(region, PreprocessConfig(patch_size=64, resize_mode=mode))
    assert patch.shape == (64, 64, 3) and patch.dtype == np.uint8


def test_pad_mode_keeps_aspect_ratio_with_replicated_borders():
    region = np.zeros((20, 80, 3), dtype=np.uint8)
    region[:, :, 0] = 200
    patch = preprocess_region(region, PreprocessConfig(patch_size=64, resize_mode="pad"))
    assert (patch[:, :, 0] == 200).all()        # bords répliqués : aucun pixel noir ajouté


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        preprocess_region(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess_region(IMAGE, PreprocessConfig(resize_mode="crop"))


def test_default_config_is_the_phase3_choice():
    assert PreprocessConfig() == PreprocessConfig(patch_size=128, margin=0.0, resize_mode="stretch")


def test_feature_dimensions_are_fixed_and_finite():
    for size in (64, 128):
        patch = region_patch(IMAGE, (10, 10, 120, 90), PreprocessConfig(patch_size=size))
        blocks = extract_feature_blocks(patch)
        assert set(blocks) == set(ALL_BLOCKS)
        assert {b: v.shape[0] for b, v in blocks.items()} == {"hog": 1764, "color": 40, "texture": 36}
        assert all(np.isfinite(v).all() and v.dtype == np.float32 for v in blocks.values())


def test_features_are_deterministic_and_respect_block_order():
    patch = region_patch(IMAGE, (0, 0, 150, 150))
    a, b = extract_features(patch), extract_features(patch.copy())
    assert np.array_equal(a, b) and a.shape == (1840,)
    only_color = extract_features(patch, FeatureConfig(blocks=("color",)))
    assert np.array_equal(only_color, a[1764:1804])


def test_uniform_patch_does_not_produce_nan():
    flat = np.full((128, 128, 3), 127, dtype=np.uint8)
    assert np.isfinite(extract_features(flat)).all()


def test_color_histograms_sum_to_one():
    v = extract_feature_blocks(region_patch(IMAGE, (0, 0, 100, 100)), FeatureConfig(blocks=("color",)))["color"]
    assert v[:18].sum() == pytest.approx(1) and v[18:26].sum() == pytest.approx(1) and v[26:34].sum() == pytest.approx(1)
