import copy

import numpy as np

from suite2p import parameters
from suite2p.gui.rungui import set_default_settings


def test_user_settings_fill_new_nested_defaults(tmp_path, monkeypatch):
    saved = parameters.default_settings()
    saved["run"].pop("roi_source")
    saved["run"].pop("sam3_enabled")
    saved["run"]["do_registration"] = 0
    np.save(tmp_path / "settings_user.npy", saved)
    monkeypatch.setattr(parameters, "SETTINGS_FOLDER", tmp_path)

    migrated = parameters.user_settings()

    assert migrated["run"]["roi_source"] == "automatic"
    assert migrated["run"]["sam3_enabled"] is False
    assert migrated["run"]["do_registration"] == 0


def test_gui_defaults_tolerate_missing_saved_keys():
    schema = copy.deepcopy(parameters.SETTINGS)
    saved = parameters.default_settings()
    saved["run"].pop("roi_source")
    saved["run"].pop("sam3_enabled")

    set_default_settings(schema, saved)

    assert schema["run"]["roi_source"]["default"] == "automatic"
    assert schema["run"]["sam3_enabled"]["default"] is False
