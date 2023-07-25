SELECT
  LISTAGG(transcript_obj:pretty_transcript, ' ') AS all_text
FROM sometable,
LATERAL FLATTEN(input => parse_json(transcript_column):transcript) transcript_obj;
