"""Dataset audit integration tests."""

from __future__ import annotations

import pytest

from industrial_defect.config import configured_path
from industrial_defect.validation import audit_dataset, render_audit_markdown


@pytest.mark.integration
def test_real_dataset_audit_is_ready(config):
    audit = audit_dataset(configured_path(config, "data_root"), str(config["data"]["category"]))
    counts = audit["counts"]
    assert counts["train_good_images"] == 213
    assert counts["test_good_images"] == 60
    assert counts["test_anomalous_images"] == 40
    assert counts["ground_truth_masks"] == 40
    assert audit["checks"]["dataset_ready"]["passed"] is True
    assert audit["provenance"]["license"] == "CC BY-NC-SA 4.0"
    assert audit["provenance"]["source"].startswith("https://www.mvtec.com/")
    critical = [issue for issue in audit["issues"] if issue["severity"] == "critical"]
    assert critical == []
    markdown = render_audit_markdown(audit)
    assert "transistor" in markdown
    assert "CC BY-NC-SA 4.0" in markdown
