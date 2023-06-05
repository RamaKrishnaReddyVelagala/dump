
import pandas as pd

# Assuming your DataFrame is named df

# Sort the DataFrame by 'len(topic)' in descending order within each 'conversation_id' group
filtered_df = df.groupby('conversation_id').apply(lambda x: x.loc[x['topic'].apply(len).idxmax()])


# Reset the index of the sorted DataFrame
sorted_df.reset_index(drop=True, inplace=True)
