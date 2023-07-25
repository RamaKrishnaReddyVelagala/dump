-- Assuming you have a table named 'your_table' with a column 'your_json_column'
WITH flattened_data AS (
  SELECT
    your_json_column:"transcript" AS transcript
  FROM
    your_table
)
SELECT
  LISTAGG(entry:"text", ' ') WITHIN GROUP (ORDER BY entry:"index") AS total_text
FROM
  flattened_data,
  LATERAL FLATTEN(input => flattened_data.transcript) entry;
