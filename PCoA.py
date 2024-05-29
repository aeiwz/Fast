import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
import plotly.express as px

# Example alpha diversity metrics DataFrame
alpha_diversity = pd.DataFrame({
    'SampleID': ['Sample1', 'Sample2', 'Sample3'],
    'Shannon': [2.3, 3.1, 2.8],
    'Simpson': [0.8, 0.7, 0.75],
    'Observed': [150, 200, 175]
})

alpha_diversity.set_index('SampleID', inplace=True)

# Calculate the distance matrix
dist_matrix = pdist(alpha_diversity, metric='euclidean')
dist_matrix_square = squareform(dist_matrix)

# Perform PCoA
pcoa_results = pcoa(dist_matrix_square)
pcoa_coords = pcoa_results.samples
pcoa_coords['SampleID'] = alpha_diversity.index

# Plot PCoA with plotly
fig = px.scatter(pcoa_coords, x=0, y=1, text='SampleID', title='PCoA of Alpha Diversity Metrics',
                 labels={'0': 'PCoA1', '1': 'PCoA2'})

# Customize the plot
fig.update_traces(marker=dict(size=12, color='blue'),
                  textposition='top right')
fig.update_layout(title='PCoA of Alpha Diversity Metrics',
                  xaxis_title='PCoA1',
                  yaxis_title='PCoA2')

# Show the plot
fig.show()





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
