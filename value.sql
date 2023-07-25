df['all_texts'] = df['json_column'].apply(lambda x: ' '.join(item['pretty_transcript'] for item in json.loads(x)['transcript']))
