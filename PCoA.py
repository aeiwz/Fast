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



