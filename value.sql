SELECT LISTAGG(transcript_value, ' ') WITHIN GROUP (ORDER BY array_index) AS concatenated_transcript
FROM (
  SELECT transcript_obj.array_index,
         transcript_obj.value:pretty_transcript::string AS transcript_value
  FROM your_table,
       LATERAL FLATTEN(input => parse_json(transcript_column), path => 'transcript') AS transcript_obj
);
