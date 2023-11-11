
"""
97 - 7 day period, today will be excluded.
"""
from grp_bde_omni_models_bde_omni_magneto_retrain.src.utils.connections import mssql_con
import pandas as pd
from pathlib import Path
from sys import path
from os.path import join
from grp_bde_omni_models_bde_omni_magneto_retrain.src.utils.helpers import \
    make_shift_starttime_endtime
from datetime import datetime, timedelta, date
import random
import numpy as np

path.insert(0, join(Path(__file__).resolve().parents[2]))

sql_dir = Path("grp_bde_omni_models_bde_omni_magneto_retrain/src/sql")

mssql_engine = mssql_con()


def read_conversationid(start_time: date, end_time: date) -> pd.DataFrame:
    """
    Read conversation_id that are human evaluated and available in SCRA tables,
    and belong to MILITARY BENEFITS.

    :param start_time: Start date for the partition_dt
    :param end_time: end date for the partition_dt(data partiiton_dt is always < end_date)
    :return: Dataframe which contains scra data and answer from the human evaluation .
    """
    with open(sql_dir / "conid_movit_stg.sql", 'r') as q:
        query = q.read().format(start_time=start_time, end_time=end_time)
    data = pd.read_sql(query, mssql_engine)

    return data


def get_max_transcript_element(data_input: pd.DataFrame) -> pd.DataFrame:
    """
    Get transcript values  for each conversation_id where the length of transcript is the highest.

    :param data_input: Data Frame that contains data from the direct input_stg and movit_stg, which has multiple duplicates for each conversation_id.
    :return: Data Frame having data that contains no duplicate conversation_id.
    """
    filtered_df = data_input.groupby('conversation_id').apply(
        lambda x: x.loc[x['transcript'].apply(len).idxmax()])
    filtered_df.reset_index(drop=True,
                            inplace=True)  # Reset the index of the sorted DataFrame
    return filtered_df


# def intime_outime_dates(filtered_df : pd.DataFrame, start_date: date, end_date:date) -> [date, date, date]:
#     """
#
#     :param filtered_df:
#     :param start_date:
#     :type start_date:
#     :param end_date:
#     :type end_date:
#     :return:
#     :rtype:
#     """
#
#     intime_start_day = filtered_df["partition_dt"].min()
#     intime_end_day = intime_start_day + timedelta(days=20)
#     final_end_day = filtered_df["partition_dt"].max()
#     print(intime_start_day, intime_end_day, final_end_day)
#
#     if start_date != intime_start_day:
#         print("Hey why is there a mis match in start dates of magneto_input_stg?")
#     if end_date != final_end_day:
#         print("Hey why is there a mis match in end dates of magneto_input_stg?")


def intime_oot_classification(filtered_df: pd.DataFrame,
                              start_date: date) -> pd.DataFrame:
    """
    Classifying values into intime and out-of-time, last 20 days are taken as oot and other 40-42 days are taken as intime

    :param filtered_df: cleaned dataframe with unqiue conv_id and maximum length on transcript.
    :param start_date:
    :return:  Pandas filtered dataframe that go a column oot_classification added on iteelf using above logic.
    """
    filtered_df["oot_classification"] = filtered_df["partition_dt"].apply(
        lambda x: "oot" if (x - start_date).days < 20 else "in-time")

    return filtered_df


def assign_split(group: pd.DataFrame) -> pd.DataFrame:
    """
    Fill/ categorize split column based on  oot-classification column
    if oot -> validation.
    if inttime -> 80% into train and other into test.
    this is done on each day, so this function is applied on dataframe after grouping it by on date-partition_dt


    :param group:  filtered_dataframe which has oot-classification column.
    :return:  filtered_dataframe has its split column added and populated.
    """
    if 'oot' in group['oot_classification'].values:
        group['split'] = 'validation'
    else:
        group_size = len(group)
        train_size = int(0.8 * group_size)
        test_size = group_size - train_size
        split = ['train'] * train_size + ['test'] * test_size
        random.shuffle(split)
        group['split'] = split
    return group


def populate_final_input_table(filtered_df: pd.DataFrame) -> None:
    """
    Insert dataframe into table: MAGNETO.BK_SCRA_RETRAINING_MAGNETO_FINAL_INPUT
    :param filtered_df: Dataframe that is ready to be inserted into the table after assigning it split.
    """
    filtered_df.to_sql(name='BK_SCRA_RETRAINING_MAGNETO_FINAL_INPUT', con=mssql_engine,
                       schema="MAGNETO",
                       if_exists='append', index=False)


def num_yes_no(filtered_df):
    no_of_yes = filtered_df[filtered_df["answer"] == '["YES"]'].shape[0]
    no_of_no = filtered_df[filtered_df["answer"] == '["NO"]'].shape[0]

    tot_scra_no = (no_of_yes * 100) - no_of_no

    return tot_scra_no, no_of_yes, no_of_no


def num_no_datewise(filtered_df, tot_scra_no, multiplier=100):
    # TODO currently we are using day by day datewise, until we have a column in
    #  input-stg, once we have it uncomment the below part and change in sql.
    date_counts = filtered_df["partition_dt"].value_counts().reset_index()
    date_counts.columns = ["date", "count"]
    date_counts["date"] = pd.to_datetime(date_counts["date"])

    # monthly_data = date_counts.groupby(date_counts['Date'].dt.to_period('M'))['Count'].sum().reset_index()
    # monthly_data.columns = ["month", "count"]

    total_count = date_counts["count"].sum()

    # the datecounts can be higher but not lower.
    date_counts["scra_count"] = ((date_counts[
                                      "count"] * tot_scra_no) / total_count).round().astype(
        int) + 1

    # return monthly_data

    return date_counts


def read_scra_output(conversation_ids, date, n):
    with open(sql_dir / "read_scra_output.sql", 'r') as q:
        query = q.read().format(conversation_ids=conversation_ids, partition_dt=date,
                                n=n)
    data = pd.read_sql(query, mssql_engine)

    return data


def intime_oot():
    start_date, end_date = make_shift_starttime_endtime()
    print(start_date, end_date)
    data_input = read_conversationid(start_date, end_date)
    filtered_df = get_max_transcript_element(data_input)
    print(filtered_df)

    # intime_outime_dates(filtered_df, start_date, end_date)
    filtered_df = intime_oot_classification(filtered_df, start_date)
    filtered_df = filtered_df.groupby('partition_dt').apply(assign_split)

    # Reset the index and drop the groupby index
    filtered_df = filtered_df.reset_index(drop=True)

    populate_final_input_table(filtered_df)

    # TODO Completed normal -, now going for scra - 100 times no.
    # TODO going for daywise-select,  concat all values and then insert one single time.
    # get how many values are there in the pandas,(for this i dodnot need a query,
    # i can get it from filtered_df.

    # Get number of yes and number of no's.
    tot_scra_no, no_of_yes, no_of_no = num_yes_no(filtered_df)
    print(tot_scra_no, no_of_yes, no_of_no)

    date_counts = num_no_datewise(filtered_df, tot_scra_no)
    print(date_counts)
    conversation_ids = filtered_df["conversation_id"].tolist()

    final_df = pd.DataFrame(columns=filtered_df.columns)
    print(final_df)
    for index, row in date_counts.iterrows():
        # if index == 1:
        #     print("index more than 1")
        #     break
        print(f"index: {index}")
        print(row[0], row[1], row[2])
        # for this date/month extract the values and keep on adding to original dataset.
        # TODO ----- how to deal with not so unqiue values?? - may be top distinct values or get more values and take.

        scra_df = read_scra_output(conversation_ids=tuple(conversation_ids),
                                   date=row[0], n=row[2])

        # TODO from here take it on.

        # filtered_df = get_max_transcript_element(data_input)

        # Reset the index and drop the groupby index
        scra_df = scra_df.reset_index(drop=True)
        scra_df["oot_classification"] = np.nan
        scra_df["split"] = np.nan
        # combining each split to final_df which will be inputted.
        final_df = pd.concat([final_df, scra_df], ignore_index=True)

    print(final_df)
    print(f"columns:{final_df.columns}")

    final_df = intime_oot_classification(final_df, start_date)
    final_df = final_df.groupby('partition_dt').apply(assign_split)
    final_df["answer"] = '["No"]'
    print(final_df)

    populate_final_input_table(final_df)
