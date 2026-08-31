from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from forecastlab.compliance import PassportComplianceEvaluator
from forecastlab.synthetic_dataset import (
    DATASET_PROVENANCE,
    DATASET_VERSION,
    synthetic_dataset_v1,
)

STYLES = """
:root {
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  color: #172033;
  background: #f5f6f8;
}
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1120px; margin: 0 auto; padding: 40px 24px 64px; }
h1 { margin: 5px 0 8px; font-size: clamp(2rem, 6vw, 4rem); }
h2 { margin: 0 0 14px; font-size: 1.15rem; }
.sub { max-width: 820px; color: #647083; line-height: 1.6; }
.note { color: #707a89; font-size: .82rem; line-height: 1.5; }
.cards {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 24px 0;
}
.card, .panel {
  background: #fff;
  border: 1px solid #dfe4ea;
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(20, 30, 50, .05);
}
.card { padding: 17px; }
.card span { display: block; color: #707a89; font-size: .75rem; text-transform: uppercase; }
.card strong { display: block; margin-top: 8px; font-size: 1.5rem; }
.panel { padding: 20px; margin-top: 18px; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: .86rem; }
th, td {
  padding: 11px 8px;
  border-bottom: 1px solid #edf0f3;
  text-align: left;
  vertical-align: top;
}
th { color: #707a89; font-weight: 600; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 999px; }
.pass { background: #dcfce7; }
.fail { background: #fee2e2; }
.rules { min-width: 390px; }
.rule { margin-bottom: 8px; line-height: 1.45; }
.score-track {
  width: 100px;
  height: 8px;
  background: #edf0f3;
  border-radius: 999px;
  overflow: hidden;
}
.score-bar { height: 100%; background: #475569; }
@media (max-width: 820px) { .cards { grid-template-columns: 1fr 1fr; } }
@media (max-width: 520px) { .cards { grid-template-columns: 1fr; } }
"""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_demo_report_html() -> str:
    evaluator = PassportComplianceEvaluator()
    dataset = synthetic_dataset_v1()
    evaluations = [(item, evaluator.evaluate(item.observation)) for item in dataset]

    compliant_count = sum(int(result.compliant) for _, result in evaluations)
    failed_rule_count = sum(
        sum(int(not rule.passed) for rule in result.rules) for _, result in evaluations
    )
    mean_score = sum(result.score for _, result in evaluations) / len(evaluations)

    rows: list[str] = []
    for item, result in evaluations:
        rule_html = "".join(
            '<div class="rule">'
            f'<span class="badge {"pass" if rule.passed else "fail"}">'
            f"{escape(rule.code.value)} · {'pass' if rule.passed else 'fail'}</span> "
            f"{escape(rule.evidence)}"
            "</div>"
            for rule in result.rules
        )
        expected = ", ".join(sorted(code.value for code in item.expected_failures)) or "none"
        rows.append(
            "<tr>"
            f"<td>{escape(item.id)}</td>"
            f"<td>{escape(item.split)}</td>"
            f"<td>{escape(expected)}</td>"
            f'<td><span class="badge {"pass" if result.compliant else "fail"}">'
            f"{'compliant' if result.compliant else 'non-compliant'}</span></td>"
            '<td><div class="score-track">'
            f'<div class="score-bar" style="width:{result.score * 100:.1f}%"></div>'
            f"</div>{_pct(result.score)}</td>"
            f'<td class="rules">{rule_html}</td>'
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForecastLab — Compliance Demo</title>
<style>{STYLES}</style>
</head>
<body>
<main>
<header>
  <div class="note">{escape(DATASET_VERSION)} · SYNTHETIC SIGNALS ONLY</div>
  <h1>Passport-photo rule evaluation</h1>
  <p class="sub">
    A deterministic browser report for the current policy layer. Each case shows the input-label
    expectation, overall decision and the evidence produced by every rule.
  </p>
</header>
<section class="cards">
  <div class="card"><span>Cases</span><strong>{len(evaluations)}</strong></div>
  <div class="card"><span>Compliant cases</span><strong>{compliant_count}</strong></div>
  <div class="card"><span>Failed rule checks</span><strong>{failed_rule_count}</strong></div>
  <div class="card"><span>Mean policy score</span><strong>{_pct(mean_score)}</strong></div>
</section>
<section class="panel">
  <h2>Rule evidence by case</h2>
  <table>
    <thead>
      <tr>
        <th>Case</th><th>Split</th><th>Expected failures</th>
        <th>Decision</th><th>Score</th><th>Rule evidence</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>
<section class="panel">
  <h2>Current boundary</h2>
  <p class="sub">
    This report evaluates precomputed observations. It does not infer pose, faces, background or
    occlusion from image pixels, and it is not an ICAO certification or a real-world accuracy claim.
  </p>
  <p class="note">{escape(DATASET_PROVENANCE)}</p>
</section>
</main>
</body>
</html>"""


def write_demo_report(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_demo_report_html(), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ForecastLab compliance demo")
    parser.add_argument("--output", default="build/forecastlab-compliance.html")
    args = parser.parse_args()
    print(write_demo_report(args.output))


if __name__ == "__main__":
    main()
