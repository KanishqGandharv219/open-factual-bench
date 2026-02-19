"""
analyze_results.py – Analyze one or more benchmark result JSON files.

Usage:
    python analyze_results.py results/run_*.json
    python analyze_results.py --html leaderboard.html results/run_*.json

Prints:
  - Per-model accuracy table
  - Per-domain accuracy breakdown
  - Hallucination stats (hallucinated vs refused)

With --html:
  - Generates a self-contained static HTML leaderboard
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime


def load_run(path: str) -> dict:
    """Load a benchmark run JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def analyze_run(data: dict, filepath: str):
    """Print detailed analysis of a single run."""
    config = data.get("config", {})
    summary = data.get("summary", {})
    results = data.get("results", [])

    model_id = config.get("model_id", "unknown")
    hardware = config.get("hardware", "unknown")

    print(f"\n{'=' * 70}")
    print(f"  Model:    {model_id}")
    print(f"  Hardware: {hardware}")
    print(f"  File:     {filepath}")
    print(f"{'=' * 70}")

    # Overall accuracy
    total = summary.get("total_tasks", len(results))
    graded = summary.get("auto_graded", 0)
    correct = summary.get("correct", 0)
    skipped = summary.get("skipped", 0)
    accuracy = summary.get("accuracy", 0)

    print(f"\n  Overall: {accuracy:.1%}  ({correct}/{graded} auto-graded, {skipped} skipped)")

    # Per-domain breakdown
    domain_stats = defaultdict(lambda: {"correct": 0, "graded": 0, "total": 0})

    # We need task domain info — try to get it from task IDs
    from bench.tasks_example import EXAMPLE_TASKS
    task_map = {t.id: t for t in EXAMPLE_TASKS}

    for r in results:
        task_id = r.get("task_id", "")
        is_correct = r.get("is_correct")
        task = task_map.get(task_id)
        domain = task.domain if task else "unknown"
        sub_domain = ""
        if task and task.metadata:
            sub_domain = task.metadata.get("sub_domain", "")

        label = sub_domain if sub_domain else domain
        domain_stats[label]["total"] += 1
        if is_correct is not None:
            domain_stats[label]["graded"] += 1
            if is_correct:
                domain_stats[label]["correct"] += 1

    print(f"\n  {'Domain':<20} {'Correct':>8} {'Graded':>8} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        acc = s["correct"] / s["graded"] if s["graded"] else 0
        print(f"  {domain:<20} {s['correct']:>8} {s['graded']:>8} {acc:>9.1%}")

    # Hallucination stats
    h_total = summary.get("hallucination_tasks", 0)
    h_hallucinated = summary.get("hallucinated", 0)
    h_refused = summary.get("refused", 0)
    h_unclear = summary.get("unclear", 0)

    if h_total > 0:
        print(f"\n  Hallucination Analysis ({h_total} tasks):")
        print(f"    Hallucinated:          {h_hallucinated}")
        print(f"    Refused/Corrected:     {h_refused}")
        print(f"    Unclear:               {h_unclear}")
    else:
        # Try to extract from per-result metadata
        h_labels = defaultdict(int)
        for r in results:
            meta = r.get("metadata") or {}
            label = meta.get("hallucination_label")
            if label:
                h_labels[label] += 1
        if h_labels:
            print(f"\n  Hallucination Analysis ({sum(h_labels.values())} tasks):")
            for label, count in sorted(h_labels.items()):
                print(f"    {label:<26} {count}")

    print()


def compare_models(runs: list):
    """Print a comparison table across multiple runs."""
    if len(runs) < 2:
        return

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    print(f"\n  {'Model':<35} {'Hardware':<18} {'Accuracy':>10} {'Halluc.':>8} {'Refused':>8}")
    print(f"  {'-'*35} {'-'*18} {'-'*10} {'-'*8} {'-'*8}")

    for filepath, data in runs:
        config = data.get("config", {})
        summary = data.get("summary", {})
        model = config.get("model_id", "unknown")
        hw = config.get("hardware", "unknown")
        acc = summary.get("accuracy", 0)
        h = summary.get("hallucinated", 0)
        r = summary.get("refused", 0)

        # Truncate long model names
        if len(model) > 33:
            model = model[:30] + "..."
        if len(hw) > 16:
            hw = hw[:13] + "..."

        print(f"  {model:<35} {hw:<18} {acc:>9.1%} {h:>8} {r:>8}")

    print()


def generate_html(runs: list, output_path: str):
    """Generate a self-contained static HTML leaderboard."""

    # Collect rows sorted by accuracy (descending)
    rows = []
    for filepath, data in runs:
        config = data.get("config", {})
        summary = data.get("summary", {})
        model = config.get("model_id", "unknown")
        hw = config.get("hardware", "unknown")
        acc = summary.get("accuracy", 0)
        correct = summary.get("correct", 0)
        graded = summary.get("auto_graded", 0)
        h = summary.get("hallucinated", 0)
        r = summary.get("refused", 0)
        # Extract date from filename or summary
        date = "unknown"
        basename = os.path.basename(filepath)
        if basename.startswith("run_") and len(basename) > 18:
            date_str = basename[4:12]  # e.g. "20260218"
            try:
                date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                pass
        rows.append({
            "model": model, "hardware": hw, "accuracy": acc,
            "correct": correct, "graded": graded,
            "hallucinated": h, "refused": r, "date": date,
        })

    rows.sort(key=lambda x: x["accuracy"], reverse=True)
    top_acc = rows[0]["accuracy"] if rows else 0

    # Build table rows HTML
    table_rows = ""
    for i, row in enumerate(rows):
        cls = ' class="top"' if row["accuracy"] == top_acc else ""
        acc_pct = f"{row['accuracy']:.1%}"
        table_rows += f"""        <tr{cls}>
          <td>{i+1}</td>
          <td><code>{row['model']}</code></td>
          <td>{acc_pct}</td>
          <td>{row['correct']}/{row['graded']}</td>
          <td>{row['hallucinated']}</td>
          <td>{row['refused']}</td>
          <td>{row['hardware']}</td>
          <td>{row['date']}</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Open Factual Bench — Leaderboard</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      padding: 2rem;
      min-height: 100vh;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 600;
      margin-bottom: 0.25rem;
      color: #f0f6fc;
    }}
    .subtitle {{
      color: #8b949e;
      font-size: 0.9rem;
      margin-bottom: 1.5rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #161b22;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    th, td {{
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid #21262d;
      font-size: 0.875rem;
    }}
    th {{
      background: #1c2128;
      color: #8b949e;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      cursor: pointer;
      user-select: none;
    }}
    th:hover {{ color: #c9d1d9; }}
    th .arrow {{ font-size: 0.6rem; margin-left: 4px; }}
    tr:hover {{ background: #1c2128; }}
    tr:nth-child(even) {{ background: #0d1117; }}
    tr.top {{ background: #1a2332; }}
    tr.top td:nth-child(3) {{
      color: #3fb950;
      font-weight: 700;
    }}
    code {{
      background: #21262d;
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-size: 0.8rem;
      color: #79c0ff;
    }}
    .footer {{
      margin-top: 1.5rem;
      color: #484f58;
      font-size: 0.8rem;
      text-align: center;
    }}
    .badge {{
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    @media (max-width: 768px) {{
      body {{ padding: 1rem; }}
      th, td {{ padding: 0.5rem; font-size: 0.8rem; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>\U0001f3af Open Factual Bench &mdash; Leaderboard</h1>
    <p class="subtitle">
      {len(rows)} model(s) evaluated on 40 factual-QA tasks (35 auto-graded + 5 hallucination stress-tests)
    </p>
    <table id="leaderboard">
      <thead>
        <tr>
          <th onclick="sortTable(0)">#<span class="arrow"></span></th>
          <th onclick="sortTable(1)">Model<span class="arrow"></span></th>
          <th onclick="sortTable(2)">Accuracy<span class="arrow"></span></th>
          <th onclick="sortTable(3)">Correct<span class="arrow"></span></th>
          <th onclick="sortTable(4)">Halluc.<span class="arrow"></span></th>
          <th onclick="sortTable(5)">Refused<span class="arrow"></span></th>
          <th onclick="sortTable(6)">Hardware<span class="arrow"></span></th>
          <th onclick="sortTable(7)">Date<span class="arrow"></span></th>
        </tr>
      </thead>
      <tbody>
{table_rows}      </tbody>
    </table>
    <p class="footer">
      Auto-generated from <code>results/index.json</code> by
      <code>analyze_results.py --html</code>
      &middot; {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </p>
  </div>
  <script>
    let sortDir = {{}};
    function sortTable(colIdx) {{
      const table = document.getElementById("leaderboard");
      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const dir = sortDir[colIdx] = !(sortDir[colIdx] || false);
      rows.sort((a, b) => {{
        let aVal = a.cells[colIdx].textContent.trim();
        let bVal = b.cells[colIdx].textContent.trim();
        let aNum = parseFloat(aVal.replace("%", ""));
        let bNum = parseFloat(bVal.replace("%", ""));
        if (!isNaN(aNum) && !isNaN(bNum)) {{
          return dir ? aNum - bNum : bNum - aNum;
        }}
        return dir ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }});
      rows.forEach(r => tbody.appendChild(r));
    }}
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nLeaderboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze open-factual-bench result files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to result JSON files",
    )
    parser.add_argument(
        "--html",
        metavar="OUTPUT",
        help="Generate a static HTML leaderboard at the given path",
    )
    args = parser.parse_args()

    runs = []
    for filepath in args.files:
        try:
            data = load_run(filepath)
            runs.append((filepath, data))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"WARNING: Could not load {filepath}: {e}", file=sys.stderr)

    if not runs:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    # Analyze each run
    for filepath, data in runs:
        analyze_run(data, filepath)

    # Compare if multiple
    compare_models(runs)

    # Generate HTML if requested
    if args.html:
        generate_html(runs, args.html)


if __name__ == "__main__":
    main()
