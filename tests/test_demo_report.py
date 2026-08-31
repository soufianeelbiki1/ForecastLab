from pathlib import Path

from forecastlab.demo_report import build_demo_report_html, write_demo_report


def test_demo_report_contains_policy_cases_and_boundaries() -> None:
    html = build_demo_report_html()

    assert "Passport-photo rule evaluation" in html
    assert "SYNTHETIC SIGNALS ONLY" in html
    assert "Rule evidence by case" in html
    assert "test-multi-failure" in html
    assert "image_dimensions" in html
    assert "background" in html
    assert "non-compliant" in html
    assert "does not infer pose, faces, background or" in html
    assert "not an ICAO certification" in html


def test_demo_report_writer_creates_standalone_html(tmp_path: Path) -> None:
    output = write_demo_report(tmp_path / "forecastlab.html")

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "<style>" in content
    assert "synthetic-observations-v1" in content
