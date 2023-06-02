
# max_dt = pd.read_sql(query, movit_mssql_engine).iloc[0, 0]
# File "/deployments/venv/lib64/python3.9/site-packages/pandas/io/sql.py", line 480, in read_sql
# pandas_sql = pandasSQL_builder(con)
# File "/deployments/venv/lib64/python3.9/site-packages/pandas/io/sql.py", line 656, in pandasSQL_builder
# return SQLDatabase(con, schema=schema, meta=meta)
# File "/deployments/venv/lib64/python3.9/site-packages/pandas/io/sql.py", line 1147, in __init__
# meta = MetaData(self.connectable, schema=schema)
# TypeError: __init__() got multiple values for argument 'schema'
