SELECT 
  LISTAGG(transcript_value, ' ') WITHIN GROUP (ORDER BY array_index) AS concatenated_transcript
FROM (
  SELECT 
    ARRAY_AGG(transcript_value) transcript_array,
    ARRAY_AGG(ARRAY_INDEX(transcript_array, transcript_value)) array_index
  FROM (
    SELECT 
      parse_json(transcript_column):"transcript" AS transcript_array
    FROM your_table
  ),
  LATERAL FLATTEN(transcript_array) transcript_value WITH OFFSET AS array_index
);
