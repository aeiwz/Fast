import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = 'path_to_your_file/Transpost-wf-16s-counts-phylum.csv'
df = pd.read_csv(file_path)

# Normalize the data to relative abundance percentages
df_percentage = df.copy()
columns_to_normalize = df.columns[3:]
df_percentage[columns_to_normalize] = df_percentage[columns_to_normalize].div(df_percentage[columns_to_normalize].sum(axis=1), axis=0) * 100

# Sum the relative abundances for each phylum across all samples
phylum_sums = df_percentage[columns_to_normalize].sum(axis=0)

# Identify the top 20 phyla based on these sums
top_20_phyla = phylum_sums.nlargest(20).index.tolist()

# Filter the dataset to include only the top 20 phyla
df_top_20 = df_percentage[['Sample_name', 'Group', 'Time point'] + top_20_phyla]

# Group the data by 'Group' and 'Time point' and compute mean
grouped = df_top_20.groupby(['Group', 'Time point']).mean().reset_index()

# Define colors for the phyla
phylum_colors = plt.get_cmap('tab20').colors

# Prepare the data for stacked bar plot
groups = grouped['Group'].unique()
time_points = grouped['Time point'].unique()
group_time_labels = [f'{group}\n{str(time_point)}' for time_point in time_points for group in groups]
bar_width = 0.35

fig, ax = plt.subplots(figsize=(15, 10))

# Plotting
for i, group in enumerate(groups):
    bottom = np.zeros(len(time_points))
    for j, phylum in enumerate(top_20_phyla):
        values = grouped[grouped['Group'] == group][phylum].values
        ax.bar(np.arange(len(time_points)) + i * bar_width, values, bar_width, bottom=bottom, color=phylum_colors[j % len(phylum_colors)], label=phylum if i == 0 else "")
        bottom += values

# Customizing plot
ax.set_title('Relative Abundance of Top 20 Bacterial Phyla by Group and Time Point (100% Scale)', fontsize=16)
ax.set_xlabel('Group and Time Point', fontsize=14)
ax.set_ylabel('Relative Abundance (%)', fontsize=14)
ax.set_xticks(np.arange(len(time_points)) + bar_width / 2)
ax.set_xticklabels(group_time_labels, rotation=90)
ax.legend(title='Phylum', bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=5)

plt.tight_layout()
plt.show()
