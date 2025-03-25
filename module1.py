import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def convert_height_to_inches(height_str):
    """
    Convert a height string in the format '6-2' to inches.
    Returns an integer (total inches) or None if conversion fails.
    """
    try:
        feet, inches = height_str.split('-')
        return int(feet) * 12 + int(inches)
    except Exception:
        return None

# Read the data
tables = pd.read_html('sportsref_download.xls')
data = tables[0]

print("First five rows of data:")
print(data.head())

# Filter the dataset to positions of interest
positions_of_interest = ['CB', 'EDGE', 'RB', 'WR']
filtered_data = data[data['Pos'].isin(positions_of_interest)].copy()

# Convert height from '6-2' to inches
filtered_data['Ht_inches'] = filtered_data['Ht'].apply(convert_height_to_inches)

# Convert relevant columns to numeric
cols_to_numeric = ['Wt', '40yd', 'Bench', 'Vertical', 'Broad Jump', '3Cone', 'Shuttle']
for col in cols_to_numeric:
    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

# Compute and round average measurables for each position
avg_stats = filtered_data.groupby('Pos').agg({
    'Ht_inches': 'mean',
    'Wt': 'mean',
    '40yd': 'mean',
    'Bench': 'mean',
    'Vertical': 'mean',
    'Broad Jump': 'mean'
}).reset_index()

# Round to desired precision
avg_stats['Ht_inches'] = avg_stats['Ht_inches'].round(2)   # Height in inches
avg_stats['Wt'] = avg_stats['Wt'].round(0)                # Weight in pounds
avg_stats['40yd'] = avg_stats['40yd'].round(2)            # 40-yard dash in seconds
avg_stats['Bench'] = avg_stats['Bench'].round(0)          # Bench reps
avg_stats['Vertical'] = avg_stats['Vertical'].round(1)    # Vertical jump in inches
avg_stats['Broad Jump'] = avg_stats['Broad Jump'].round(0)# Broad jump in inches

print("\nAverage stats by position (rounded):")
print(avg_stats)


# This chart can reveal how weight correlates with speed and whether heavier players at certain positions run comparably fast or slow relative to others.

sns.set_theme(style="whitegrid")

# Create a regression plot (scatter + best-fit line) for weight vs 40-yard dash, color coded by position.
g = sns.lmplot(
    data=filtered_data, x="Wt", y="40yd", hue="Pos",
    height=5, aspect=1.4, scatter_kws={'alpha': 0.6}
)

# Adjust labels and title
g.set_axis_labels("Weight (lbs)", "40-Yard Dash (seconds)")
plt.title("Relationship between Weight and 40-Yard Dash by Position")

plt.show()
