import os
import pandas as pd
import datetime

# CONFIGURATION
METRICS_PATH = "./Data/weekly_metrics.csv"  # or .json
VISUALS_DIR = "./figures"
OUTPUT_DIR = "./summaries"
APP_NAME = "Movie Comparer"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(METRICS_PATH, exist_ok=True)

def load_metrics(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".json"):
        return pd.read_json(path)
    else:
        raise ValueError("Unsupported metrics format")

def generate_markdown(metrics_df, image_files):
    date_str = datetime.date.today().strftime("%B %d, %Y")
    md = f"# {APP_NAME} Weekly Summary\n\n"
    md += f"**Date:** {date_str}\n\n"
    md += "## 📊 Model Metrics\n\n"
    md += metrics_df.to_markdown(index=False)
    md += "\n\n## 🖼️ Visuals\n"
    for img in image_files:
        md += f"![{img}]({VISUALS_DIR}{img})\n"
    return md

def generate_html(metrics_df, image_files):
    date_str = datetime.date.today().strftime("%B %d, %Y")
    html = f"""<html>
<head><title>{APP_NAME} Weekly Summary</title></head>
<body>
    <h1>{APP_NAME} Weekly Summary</h1>
    <p><strong>Date:</strong> {date_str}</p>
    <h2>📊 Model Metrics</h2>
    {metrics_df.to_html(index=False)}
    <h2>🖼️ Visuals</h2>
"""
    for img in image_files:
        html += f'<img src="{VISUALS_DIR}{img}" alt="{img}" style="max-width:600px;"><br>\n'
    html += "</body>\n</html>"
    return html

def main():
    try:
        metrics_df = load_metrics(METRICS_PATH)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if metrics_df.empty:
        print("Error: metrics file is empty. Exiting.")
        sys.exit(1)

    # list PNG files robustly (case-insensitive)
    try:
        image_files = [f for f in os.listdir(VISUALS_DIR) if f.lower().endswith(".png")]
    except Exception as e:
        print(f"Error listing visuals directory: {e}")
        image_files = []

    markdown = generate_markdown(metrics_df, image_files)
    html = generate_html(metrics_df, image_files)

    week_id = datetime.date.today().strftime("%Y-%m-%d")
    md_path = os.path.join(OUTPUT_DIR, f"summary_{week_id}.md")
    html_path = os.path.join(OUTPUT_DIR, f"summary_{week_id}.html")

    try:
        with open(md_path, "w", encoding="utf-8") as f_md:
            f_md.write(markdown)
        with open(html_path, "w", encoding="utf-8") as f_html:
            f_html.write(html)
    except Exception as e:
        print(f"Error writing output files: {e}")
        sys.exit(1)

    print(f"✅ Weekly summary generated: {os.path.basename(md_path)} and {os.path.basename(html_path)}")

if __name__ == "__main__":
    main()
