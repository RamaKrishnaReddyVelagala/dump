-- Assuming you have a table named 'your_table' with a column 'your_json_column'
SELECT
  LISTAGG(entry_value:"text", ' ') WITHIN GROUP (ORDER BY entry_key) AS total_text
FROM (
  SELECT
    entry.key AS entry_key,
    entry.value AS entry_value
  FROM
    your_table,
    LATERAL FLATTEN(input => your_json_column:"transcript") entry
)
GROUP BY 1;
