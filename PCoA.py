import pandas as pd
import numpy as np
from skbio.stats.ordination import pcoa
from skbio.diversity import beta_diversity
import plotly.express as px
import plotly.io as pio

# Set default renderer to open the plot in the browser
pio.renderers.default = 'browser'

# Load the dataset
file_path = 'path/to/your/Transpose-wf-16s-diversity.csv'  # Update with your file path
data = pd.read_csv(file_path, index_col=0)

# Ensure all data is numeric, coerce errors to NaN and then fill NaNs with 0 (or another strategy)
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Assuming your data is in the form of samples (rows) and features (columns)
# If you have OTU or ASV table, you might need to transpose it
# data = data.T 

# Calculate distance matrix (example using Bray-Curtis distance)
distance_matrix = beta_diversity('braycurtis', data.values, data.index)

# Perform PCoA
pcoa_results = pcoa(distance_matrix)

# Create a DataFrame for the PCoA results
pcoa_df = pcoa_results.samples.reset_index()

# Rename columns for Plotly
pcoa_df.columns = ['SampleID', 'PC1', 'PC2', 'PC3'] + list(pcoa_df.columns[4:])

# Calculate the proportion of variance explained by each principal coordinate
var_explained = pcoa_results.proportion_explained * 100

# Plot the results using Plotly
fig = px.scatter(pcoa_df, x='PC1', y='PC2', text='SampleID',
                 labels={
                     'PC1': f'PC1 ({var_explained[0]:.2f}%)',
                     'PC2': f'PC2 ({var_explained[1]:.2f}%)'
                 },
                 title='PCoA Plot')

# Add annotations for each point
fig.update_traces(textposition='top center')

# Show the plot
fig.show()





import numpy as np
import plotly.express as px
from sklearn.manifold import MDS

# Sample data creation
# Note: Replace this sample data with actual data if available.
# Simulating data structure for demonstration
categories = ['No-take', 'Open fishing', 'Limited access']
management_status = ['Before management', 'After management']
points_per_category = 10

np.random.seed(42)
data = []

for status in management_status:
    for category in categories:
        for _ in range(points_per_category):
            x, y = np.random.normal(loc=0.0, scale=0.5, size=2)
            data.append([x, y, category, status])

data = np.array(data)

# Performing NMDS
nmds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
nmds_fit = nmds.fit_transform(data[:, :2].astype(float))

# Convert to a DataFrame for Plotly
import pandas as pd

df = pd.DataFrame(nmds_fit, columns=['NMDS1', 'NMDS2'])
df['Category'] = data[:, 2]
df['Status'] = data[:, 3]

# Plotting with Plotly
fig = px.scatter(df, x='NMDS1', y='NMDS2', color='Category', symbol='Status',
                 title='NMDS Plot Before and After Management',
                 labels={'NMDS1': 'NMDS1', 'NMDS2': 'NMDS2'},
                 category_orders={'Category': categories, 'Status': management_status})

fig.show()


import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go

# Filter the dataframe for the relevant time points
relevant_time_points = [f'Day {i}' for i in range(-1, 15)]
filtered_df = df[df['Time point'].isin(relevant_time_points)]

# Get the unique time points
unique_time_points = filtered_df['Time point'].unique()

# Create a subplot figure
fig = sp.make_subplots(rows=5, cols=3, subplot_titles=unique_time_points, vertical_spacing=0.1, horizontal_spacing=0.1)

# Define the layout for subplots
rows = 5
cols = 3
current_row = 1
current_col = 1

# Define the colors for different groups
group_colors = {
    'Cr': 'blue',
    'Other': 'green'
}

# Loop through each unique time point to create subplots
for time_point in unique_time_points:
    # Filter data for the current time point
    time_point_df = filtered_df[filtered_df['Time point'] == time_point]
    
    # Melt the dataframe for Plotly Express
    melted_df = time_point_df.melt(id_vars=['Sample_name', 'Group', 'Time point'], var_name='Phylum', value_name='Count')
    
    # Create a bar plot for the current time point
    fig_data = []
    for group in time_point_df['Group'].unique():
        group_df = melted_df[melted_df['Group'] == group]
        fig_data.append(
            go.Bar(
                x=group_df['Phylum'],
                y=group_df['Count'],
                name=group,
                marker_color=group_colors.get(group, 'red'),
                showlegend=(current_row == 1 and current_col == 1)  # Show legend only for the first subplot
            )
        )
    
    # Add the subplot to the figure
    for trace in fig_data:
        fig.add_trace(trace, row=current_row, col=current_col)
    
    # Update the subplot location
    if current_col < cols:
        current_col += 1
    else:
        current_col = 1
        current_row += 1

# Update layout to make it more readable
fig.update_layout(
    title_text="Phylum Counts Across Different Time Points",
    showlegend=True,
    height=1500
)

fig.show()




