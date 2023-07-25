SELECT LISTAGG(transcript_value, ' ') WITHIN GROUP (ORDER BY array_index) AS concatenated_transcript
FROM (
  SELECT 
    ARRAY_INDEX(transcript_array, transcript_value) AS array_index,
    transcript_value:pretty_transcript::string AS transcript_value
  FROM your_table,
  LATERAL FLATTEN(input => parse_json(transcript_column), OUTER => true) transcript_array
);
