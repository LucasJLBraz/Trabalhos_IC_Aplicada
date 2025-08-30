#!/usr/bin/env python3
"""
CSV → Markdown table converter.

Usage:
  py convert.py path/to/file.csv
  python convert.py path/to/file.csv
"""

import argparse
import csv
import io
import os
import sys

def read_csv(path: str) -> list[list[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()

    if raw.strip() == "":
        return []

    # Try to detect delimiter
    try:
        dialect = csv.Sniffer().sniff(raw[:8192], delimiters=[",",";","\t","|",":"])
        delim = dialect.delimiter
    except Exception:
        delim = ","

    reader = csv.reader(io.StringIO(raw), delimiter=delim)
    return [row for row in reader]

def escape_md(s: str) -> str:
    return s.replace("|", r"\|").replace("\n", "<br>")

def pad_table(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]

def to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    rows = pad_table(rows)
    header, *body = rows

    # Escape markdown chars
    header = [escape_md(str(x)) for x in header]
    body = [[escape_md(str(x)) for x in row] for row in body]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to Markdown (same name .md file).")
    parser.add_argument("csvfile", help="Path to CSV file")
    args = parser.parse_args()

    rows = read_csv(args.csvfile)
    if not rows:
        print("Error: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    md_text = to_markdown(rows)

    base, _ = os.path.splitext(args.csvfile)
    outfile = base + ".md"

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Markdown table written to: {outfile}")

if __name__ == "__main__":
    main()

