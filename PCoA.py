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
