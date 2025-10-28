import os
import pandas as pd
import datetime
from jinja2 import Template

# CONFIGURATION
METRICS_PATH = "data/weekly_metrics.csv"  # or .json
VISUALS_DIR = "visuals/"
OUTPUT_DIR = "summaries/"
APP_NAME = "MovieMatch"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    html_template = Template("""
    <html>
    <head><title>{{ app_name }} Weekly Summary</title></head>
    <body>
        <h1>{{ app_name }} Weekly Summary</h1>
        <p><strong>Date:</strong> {{ date }}</p>
        <h2>📊 Model Metrics</h2>
        {{ metrics_html | safe }}
        <h2>🖼️ Visuals</h2>
        {% for img in images %}
            <img src="{{ visuals_dir }}{{ img }}" alt="{{ img }}" style="max-width:600px;"><br>
        {% endfor %}
    </body>
    </html>
    """)
    return html_template.render(
        app_name=APP_NAME,
        date=date_str,
        metrics_html=metrics_df.to_html(index=False),
        images=image_files,
        visuals_dir=VISUALS_DIR
    )

def main():
    metrics_df = load_metrics(METRICS_PATH)
    image_files = [f for f in os.listdir(VISUALS_DIR) if f.endswith(".png")]

    markdown = generate_markdown(metrics_df, image_files)
    html = generate_html(metrics_df, image_files)

    week_id = datetime.date.today().strftime("%Y-%m-%d")
    with open(f"{OUTPUT_DIR}summary_{week_id}.md", "w") as f_md:
        f_md.write(markdown)
    with open(f"{OUTPUT_DIR}summary_{week_id}.html", "w") as f_html:
        f_html.write(html)

    print(f"✅ Weekly summary generated: summary_{week_id}.md and .html")

if __name__ == "__main__":
    main()

