
import pandas as pd

# Assuming your DataFrame is named df

# Sort the DataFrame by 'len(topic)' in descending order within each 'conversation_id' group
sorted_df = df.groupby('conversation_id', as_index=False).apply(lambda x: x.nlargest(1, 'topic', key=lambda x: len(x['topic'])))

# Reset the index of the sorted DataFrame
sorted_df.reset_index(drop=True, inplace=True)
