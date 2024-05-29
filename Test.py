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




import pandas as pd

# Load the CSV file to see its contents
file_path = '/mnt/data/Transpost-wf-16s-counts-genus.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get unique time points and groups
time_points = df['Time point'].unique()
groups = df['Group'].unique()

# Initialize the subplot figure
fig = make_subplots(rows=1, cols=4, subplot_titles=time_points)

# Generate stacked bar plots for each time point
for idx, time_point in enumerate(time_points, start=1):
    # Filter the dataframe for the current time point
    df_time = df[df['Time point'] == time_point]
    
    # Sum the counts for each group
    group_sums = df_time.groupby('Group').sum().reset_index()
    
    # Add a stacked bar trace for each group
    for col in group_sums.columns[1:]:
        fig.add_trace(
            go.Bar(
                x=group_sums['Group'],
                y=group_sums[col],
                name=col,
                showlegend=(idx == 1)  # Show legend only for the first subplot
            ),
            row=1, col=idx
        )

# Update layout
fig.update_layout(
    title_text="Alpha Abundance Stacked Bar Plot by Time Point",
    barmode='stack'
)

# Show the plot
fig.show()


# Fix the warning by selecting only numeric columns for aggregation
# Re-initialize the subplot figure
fig = make_subplots(rows=1, cols=4, subplot_titles=time_points)

# Generate stacked bar plots for each time point
for idx, time_point in enumerate(time_points, start=1):
    # Filter the dataframe for the current time point
    df_time = df[df['Time point'] == time_point]
    
    # Sum the counts for each group, considering only numeric columns
    group_sums = df_time.groupby('Group').sum(numeric_only=True).reset_index()
    
    # Add a stacked bar trace for each group
    for col in group_sums.columns[1:]:
        fig.add_trace(
            go.Bar(
                x=group_sums['Group'],
                y=group_sums[col],
                name=col,
                showlegend=(idx == 1)  # Show legend only for the first subplot
            ),
            row=1, col=idx
        )

# Update layout
fig.update_layout(
    title_text="Alpha Abundance Stacked Bar Plot by Time Point",
    barmode='stack'
)

# Show the plot
fig.show()



#100% scale

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the CSV file
file_path = 'path_to_your_file/Transpost-wf-16s-counts-genus.csv'
df = pd.read_csv(file_path)

# Get unique time points and groups
time_points = df['Time point'].unique()
groups = df['Group'].unique()

# Initialize the subplot figure
fig = make_subplots(rows=1, cols=4, subplot_titles=time_points)

# Generate stacked bar plots for each time point
for idx, time_point in enumerate(time_points, start=1):
    # Filter the dataframe for the current time point
    df_time = df[df['Time point'] == time_point]
    
    # Sum the counts for each group, considering only numeric columns
    group_sums = df_time.groupby('Group').sum(numeric_only=True).reset_index()
    
    # Normalize the counts to percentages
    group_sums.iloc[:, 1:] = group_sums.iloc[:, 1:].div(group_sums.iloc[:, 1:].sum(axis=1), axis=0) * 100
    
    # Add a stacked bar trace for each group
    for col in group_sums.columns[1:]:
        fig.add_trace(
            go.Bar(
                x=group_sums['Group'],
                y=group_sums[col],
                name=col,
                showlegend=(idx == 1)  # Show legend only for the first subplot
            ),
            row=1, col=idx
        )

# Update layout
fig.update_layout(
    title_text="100% Stacked Bar Plot of Alpha Abundance by Time Point",
    barmode='stack'
)

# Show the plot
fig.show()

# top 20

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the CSV file
file_path = 'path_to_your_file/Transpost-wf-16s-counts-genus.csv'
df = pd.read_csv(file_path)

# Sum the abundance counts for each genus across all samples
genus_sums = df.drop(columns=['Group', 'Time point']).sum().sort_values(ascending=False)

# Identify the top 20 most abundant genera
top_20_genera = genus_sums.head(20).index

# Filter the dataset to include only the top 20 genera
df_top20 = df[['Group', 'Time point'] + list(top_20_genera)]

# Get unique time points and groups
time_points = df['Time point'].unique()
groups = df['Group'].unique()

# Initialize the subplot figure
fig = make_subplots(rows=1, cols=4, subplot_titles=time_points)

# Generate 100% stacked bar plots for each time point
for idx, time_point in enumerate(time_points, start=1):
    # Filter the dataframe for the current time point
    df_time = df_top20[df_top20['Time point'] == time_point]
    
    # Sum the counts for each group, considering only numeric columns
    group_sums = df_time.groupby('Group').sum(numeric_only=True).reset_index()
    
    # Normalize the counts to percentages
    group_sums.iloc[:, 1:] = group_sums.iloc[:, 1:].div(group_sums.iloc[:, 1:].sum(axis=1), axis=0) * 100
    
    # Add a stacked bar trace for each group
    for col in group_sums.columns[1:]:
        fig.add_trace(
            go.Bar(
                x=group_sums['Group'],
                y=group_sums[col],
                name=col,
                showlegend=(idx == 1)  # Show legend only for the first subplot
            ),
            row=1, col=idx
        )

# Update layout
fig.update_layout(
    title_text="100% Stacked Bar Plot of Top 20 Alpha Abundance by Time Point",
    barmode='stack'
)

# Show the plot
fig.show()


#Top 20 fix colors
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load the CSV file
file_path = 'path_to_your_file/Transpost-wf-16s-counts-genus.csv'
df = pd.read_csv(file_path)

# Sum the abundance counts for each genus across all samples
genus_sums = df.drop(columns=['Group', 'Time point']).sum().sort_values(ascending=False)

# Identify the top 20 most abundant genera
top_20_genera = genus_sums.head(20).index

# Filter the dataset to include only the top 20 genera
df_top20 = df[['Group', 'Time point'] + list(top_20_genera)]

# Get unique time points and groups
time_points = df['Time point'].unique()
groups = df['Group'].unique()

# Define a color palette
color_palette = px.colors.qualitative.Plotly

# Assign colors to each genus
genus_colors = {genus: color_palette[i % len(color_palette)] for i, genus in enumerate(top_20_genera)}

# Initialize the subplot figure
fig = make_subplots(rows=1, cols=4, subplot_titles=time_points)

# Generate 100% stacked bar plots for each time point
for idx, time_point in enumerate(time_points, start=1):
    # Filter the dataframe for the current time point
    df_time = df_top20[df_top20['Time point'] == time_point]
    
    # Sum the counts for each group, considering only numeric columns
    group_sums = df_time.groupby('Group').sum(numeric_only=True).reset_index()
    
    # Normalize the counts to percentages
    group_sums.iloc[:, 1:] = group_sums.iloc[:, 1:].div(group_sums.iloc[:, 1:].sum(axis=1), axis=0) * 100
    
    # Add a stacked bar trace for each group
    for col in group_sums.columns[1:]:
        fig.add_trace(
            go.Bar(
                x=group_sums['Group'],
                y=group_sums[col],
                name=col,
                marker_color=genus_colors[col],
                showlegend=(idx == 1)  # Show legend only for the first subplot
            ),
            row=1, col=idx
        )

# Update layout
fig.update_layout(
    title_text="100% Stacked Bar Plot of Top 20 Alpha Abundance by Time Point",
    barmode='stack'
)

# Show the plot
fig.show()
