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
