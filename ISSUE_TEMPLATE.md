## Issue: Feature Importance of Tuned Model Plotly Code

### Description
Create a Plotly horizontal bar chart to visualize feature importance from the tuned model.

### Location
**File:** `notebooks/eda.ipynb`

### Current Implementation
The code creates a feature importance visualization with:
- Horizontal bar chart showing feature importance
- Features filtered to only show importance > 0
- Steelblue color scheme
- Proper layout with margins and height settings
- Output saved to `figures/improved/feature_importance_tuned_model.png`

### Code
```python
import plotly.express as px
import pandas as pd

# Create DataFrame of feature importances
feat_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
})

# Filter to only features with importance > 0
feat_df = feat_df[feat_df['Importance'] > 0].sort_values(by='Importance', ascending=True)

# Create Plotly horizontal bar chart
fig = px.bar(
    feat_df,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Feature Importance of Tuned Model',
    labels={'Importance': 'Importance', 'Feature': 'Feature'},
    color_discrete_sequence=['steelblue']
)

# Improve layout
fig.update_layout(
    xaxis_title='Importance',
    yaxis_title='Feature',
    margin=dict(l=100, r=20, t=50, b=50),
    height=600
)

# Show the figure
fig.show()
```

### Related Files
- `apputil.py` - Contains `train_baseline()` function
- `Data/processed/movies_model.csv` - Data file

### Dependencies
- plotly
- pandas
- scikit-learn (for model training)

### Notes
- The code depends on `train_baseline()` from `apputil.py` which returns a trained model and feature columns
- Features are sorted in ascending order for better horizontal bar chart readability
- Only features with importance > 0 are displayed to reduce clutter
