"""Tests for main.py's config loading/validation."""
import pytest
import yaml

from main import load_config, validate_config, REQUIRED_CONFIG_KEYS, REQUIRED_HYPERPARAM_GRID_KEYS, REQUIRED_TRAINING_KEYS


def test_real_config_loads_and_validates():
    cfg = load_config('configs/benchmark.yaml')
    assert REQUIRED_CONFIG_KEYS.issubset(cfg.keys())
    assert REQUIRED_HYPERPARAM_GRID_KEYS.issubset(cfg['hyperparam_grid'].keys())
    assert REQUIRED_TRAINING_KEYS.issubset(cfg['training'].keys())


@pytest.fixture
def valid_cfg():
    return {
        'sfreq': 200, 'freq_bands': ['all'], 'need_preprocessing': False,
        'k_folds': 5, 'n_repeats': 10, 'run_label': 'test',
        'hyperparam_grid': {'batch_sizes': [16], 'learning_rates': [0.001], 'l2_weight_decays': [0.0]},
        'training': {
            'patience': 30, 'full_train_max_epochs': 100, 'pilot_max_epochs': 30,
            'val_batch_size': 64, 'test_batch_size': 128, 'subsample_fraction': 0.1,
            'tsne_n_iter': 3000, 'tsne_random_state': 42,
        },
    }


def test_validate_config_accepts_well_formed_config(valid_cfg):
    assert validate_config(valid_cfg, 'in-memory') == valid_cfg


def test_validate_config_rejects_missing_top_level_key(valid_cfg):
    del valid_cfg['run_label']
    with pytest.raises(ValueError, match="run_label"):
        validate_config(valid_cfg, 'in-memory')


def test_validate_config_rejects_missing_hyperparam_grid_key(valid_cfg):
    del valid_cfg['hyperparam_grid']['learning_rates']
    with pytest.raises(ValueError, match="learning_rates"):
        validate_config(valid_cfg, 'in-memory')


def test_validate_config_rejects_missing_training_key(valid_cfg):
    del valid_cfg['training']['patience']
    with pytest.raises(ValueError, match="patience"):
        validate_config(valid_cfg, 'in-memory')


def test_load_config_broken_yaml_fails_fast_with_named_key(tmp_path, valid_cfg):
    del valid_cfg['training']['patience']
    broken_path = tmp_path / 'broken.yaml'
    broken_path.write_text(yaml.safe_dump(valid_cfg))

    with pytest.raises(ValueError, match="patience"):
        load_config(str(broken_path))