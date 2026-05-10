"""Smoke tests. Must pass on every supported platform with the base install."""

import importlib


def test_package_imports():
    mod = importlib.import_module("riprap_models")
    assert hasattr(mod, "__version__")


def test_cli_module_imports():
    cli = importlib.import_module("riprap_models.cli")
    assert hasattr(cli, "main")


def test_submodules_import():
    for name in [
        "riprap_models.device",
        "riprap_models.energy",
        "riprap_models.common.metrics",
        "riprap_models.common.provenance",
    ]:
        importlib.import_module(name)
