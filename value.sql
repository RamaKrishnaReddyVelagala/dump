SELECT LISTAGG(value, ' ') WITHIN GROUP (ORDER BY array_index) AS concatenated_transcript
FROM (
  SELECT transcript.value AS value, array_index
  FROM your_table,
  LATERAL FLATTEN(input => parse_json(transcript_column), path => 'transcript')
);
