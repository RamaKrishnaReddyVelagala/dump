import pickle
import xgboost as xgb
import json
from datetime import datetime
from sqlalchemy.sql import text
from scipy import stats
print("XGBoost version install: {}".format(xgb.__version__))
import nltk

print("NLTK version installed: {}".format(nltk.__version__))
from grp_bde_omni_models_bde_omni_magneto_retrain.src.utils.helpers import config_parser
from nltk.stem.porter import *
# import spacy
# print("SpaCy version installed: {}".format(spacy.__version__))
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score
from grp_bde_omni_models_bde_omni_magneto_retrain.src.utils.connections import mssql_con
from grp_bde_omni_models_bde_omni_magneto_retrain.src.utils.logging_config import logger
from grp_bde_omni_models_bde_omni_magneto_retrain.src.utils.helpers import \
    make_shift_starttime_endtime, read_sql_file
import pandas as pd
import xgboost
import ast
from typing import Tuple, Type

start_date, end_date = make_shift_starttime_endtime()
logger.info(
    f"Date range for the given Magneto RUN. start_date={start_date}, end_date<{end_date}")

mssql_engine = mssql_con()  # todo rework this part.
sandbox_engine = mssql_con(run_env="SANDBOX")

# Create config object and read local local_config.ini file
local_config = config_parser(file_name="local_config.ini")
config = config_parser(file_name="config.ini")

import warnings

warnings.filterwarnings('ignore')


def read_data_finalinput() -> pd.DataFrame:
    """
    Read  data from magneto.BK_SCRA_RETRAINING_MAGNETO_FINAL_INPUT with lies in dates provided.

    :return: Pandas data frame containing data from table:BK_SCRA_RETRAINING_MAGNETO_FINAL_INPUT
    """

    logger.info("Starting read_data_finalinput")
    sql = read_sql_file("read_final_input.sql")
    try:
        df_data = pd.read_sql(sql.format(start_time=start_date, end_time=end_date),
                              mssql_engine)
    except Exception as e:
        logger.info(
            "Not able to read data from table Magneto.BK_SCRA_RETRAINING_MAGNETO_FINAL_INPUT- MSSQL")
        raise Exception(e)
    logger.info("Exiting read_data_finalinput")
    return df_data


def convert_yes_no_to_binary(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert '["YES"]' and '["NO"]' into 1's and 0's and ignoring other values in the answer column.

    :param df_data:
    :return:the initial dataframe where its answer is now a binary value of 1 and o's
    """
    logger.info("Starting convert_yes_no_to_binary.")

    replacement_dict = {'["NO"]': 0, '["YES"]': 1}

    # There are some None and NA values which we want to ignore.
    valid_answer_values = list(replacement_dict.keys())
    df_data = df_data[df_data['answer'].isin(valid_answer_values)]

    # The answer columns contains majorly stringified forms of yes and no.
    df_data['answer'] = df_data['answer'].map(replacement_dict)

    logger.info("Exiting convert_yes_no_to_binary.")
    return df_data


def df_oot_intime(df_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splitting all the data in final input table into oot and intime dataframes.

    :param df_data:  A dataframe containing all the data from magneto.input table.
    :return:  2 dataframes splitting into intime and oot dataframes.
    """
    logger.info("Starting df_oot_intime")

    df_oot = df_data[df_data["split"] == "validation"]
    df_intime = df_data[df_data["split"].isin(["train", "test"])]

    logger.info("Starting df_oot_intime")
    return df_oot, df_intime


def intime_test_train_split(df_intime) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splitting data into x, Y train and test for further

    :param df_intime: data frame which has only intime.
    :return: return 4 sets of dataframe x -train,test, y-train-test respectively.
    """
    logger.info("Starting intime_test_train_split.")

    y_train, y_test = df_intime[df_intime["split"] == "train"]["answer"], \
                      df_intime[df_intime["split"] == "test"]["answer"]
    X_train, X_test = df_intime[df_intime["split"] == "train"], df_intime[
        df_intime["split"] == "test"]

    # dropping answer from X, as target values shouldn't be present training.
    X_train, X_test = X_train.drop("answer", axis=1), X_test.drop("answer", axis=1)

    logger.info("Starting intime_test_train_split.")
    return X_train, X_test, y_train, y_test


def initialize_vectorizer() -> Type[CountVectorizer]:
    """
    Initialzie count vectorizier with 70,000 features vocabulary from nlp-terms in cofnig.

    :return: count vectorizer.
    """
    logger.info("Started initialize_vectorizer.")
    nlp_terms = eval(config['NLP']['nlp_terms'])
    logger.info(f"nlp_terms: {nlp_terms}")

    # vectorize tokens for nlp_terms (features)
    logger.debug("Vectorizing tokens for NLP terms")
    vectorizer = CountVectorizer(max_features=70000,
                                 analyzer="word",
                                 lowercase=False,
                                 tokenizer=lambda doc: doc,
                                 ngram_range=(1, 1),
                                 vocabulary=nlp_terms,
                                 min_df=0.0)
    # column_names = vectorizer.vocabulary

    logger.info("Exiting initialize_vectorizer.")
    return vectorizer


def load_champion_model(MODEL_PATHS='MODEL_PATHS') -> Type[
    xgboost.sklearn.XGBClassifier]:
    """
    Load champion from model path.

    :param MODEL_PATHS: model_path directories where the model is loaded.
    :return: a xgboost- model which was already loaded.
    """

    logger.info("Starting load_champion_model.")

    # parse config file to load path to champion model pickle
    model_path = config[MODEL_PATHS]['champion_model_path']
    logger.debug(f"model_path: {model_path}")

    # load champion pickle
    with open(model_path, 'rb') as fmodel:
        champion_model = pickle.load(fmodel)

    logger.info("Exiting load_champion_model.")
    return champion_model


def read_hyperparameters() -> dict:
    """
    Read Hyperparameters from config-file.

    :return: dictionary containing hyper parameters.
    """
    logger.info("Starting read_hyperparameters.")

    # Read params and assign to a dictionary , eval on values to interpret strings to python expressions.
    params = {key: eval(value) for key, value in config['XGBOOST'].items()}

    logger.debug(f"params: {params}")
    logger.info("Exiting read_hyperparameters.")
    return params


def select_best_params(X_train_transformed, y_train, challenger_model: Type[xgboost.sklearn.XGBClassifier], params:dict, stratified_kfold: Type[StratifiedKFold]) -> dict:
    """

    :param X_train_transformed: Contains preprocessed and transformed input_features.
    :param y_train: target or output values corresponding to the training data.
    :param challenger_model: an xgboost.sklearn.XGBClassifier
    :param params: dicitonary that contains parameters.
    :param stratified_kfold:
    :return:
    """
    # create randomized cross validation search
    # we can go with either n_iter==300 or n_iter==500 depending on workspace resources
    random_search = RandomizedSearchCV(challenger_model, param_distributions=params,
                                       n_iter=10, scoring='balanced_accuracy',
                                       cv=stratified_kfold, n_jobs=-1, verbose=1)

    # execute randomized 5-fold stratified cross validation with 500 iterations which
    # will result in 2,500 different parameter combinations
    random_search.fit(X_train_transformed, y_train.to_frame()["answer"])

    # #### Select best parameters from RandomizedSearchCV and update Challenger Model
    # grab the best parameters from the randomized search
    best_params = random_search.best_params_

    return best_params


def winner(acc_champion, acc_challenger) -> bool:
    print(f"Champion model balanced accuracy: {acc_champion:.4f}")

    print(f"Challenger model balanced accuracy: {acc_challenger:.4f}")
    threshold = config['common']['champion_threshold']
    if acc_challenger > eval(threshold) * acc_champion:
        print("Challenger is the winner.")
        return 1
    else:
        print("Champion is the winner.")
        return 0


def config_to_json() -> str:
    config_json = {section: dict(config.items(section)) for section in
                   config.sections()}
    single_quote = "'"
    double_single_quote = "''"
    config_json_str = json.dumps(config_json).replace(single_quote, double_single_quote)
    return config_json_str


def insert_magneto_pickles(challenger_pickle, champion_pickle, daterange, winner_flag,
                           odate):

    # not adding challner_pickle and champion pickle in format, because they are
    # binary instead directly passing to sql at execution as parameters.
    pickle_sql = read_sql_file("insert_magneto_pickle.sql").format(
        daterange=daterange,
        winner_flag=winner_flag,
        odate=odate)

    try:
        with sandbox_engine.connect() as connection:
            connection.execute(text(pickle_sql),
                               {"data1": challenger_pickle, "data2": champion_pickle})
    except Exception as e:
        # todo this file appendix must be changed.
        # print(e[:1000])
        with open("error.txt", "w") as file:
            file.write(str(e))


def insert_magneto_summary_table(winner_flag, config_json_str, challenger_pickle,
                                 champion_pickle):
    daterange = f"{start_date} - {end_date}"
    odate = datetime.now()
    output_sql = read_sql_file("insert_magneto_output.sql").format(daterange=daterange,
                                                                   winner_flag=winner_flag,
                                                                   config_ini=config_json_str,
                                                                   odate=odate)
    try:
        pd.read_sql(output_sql, sandbox_engine)
    except Exception as e:
        print(e)
    insert_magneto_pickles(daterange=daterange, winner_flag=winner_flag, odate=odate,
                           challenger_pickle=challenger_pickle,
                           champion_pickle=champion_pickle)


# todo clean all_composing function thoroughly.
def all_composing():
    # Read data from magneto final_input table for given date-ranges.
    df_data = read_data_finalinput()

    # convert target column values to int from string.
    df_data = convert_yes_no_to_binary(df_data=df_data)

    # use ast.literal_eval to convert string representation of list to actual list of strings
    df_data['mbr_ngrams'] = df_data['transcript'].apply(ast.literal_eval)

    # fetch intime and oot into seperate dataframes.
    df_oot, df_intime = df_oot_intime(df_data)

    X_train, X_test, y_train, y_test = intime_test_train_split(df_intime)

    # print(f"X_train: {X_train}")
    # print(f"X_test: {X_test}")
    # print(f"y_train: {y_train}")
    # print(f"y_test: {y_test}")

    # loading NLP terms and initialize count_vectorizer
    vectorizer = initialize_vectorizer()

    # TODO - needs work from here.
    # fit transform the in-time train dataset that will be used to re-train challenger model.
    A_train = vectorizer.fit_transform(X_train['mbr_ngrams'])

    # In[13]:

    # transform the in-time test and out-of-time dataset that will be used to compare performance of champion vs challenger
    A_test = vectorizer.transform(X_test['mbr_ngrams'])
    A_oot = vectorizer.transform(df_oot['mbr_ngrams'])

    # convert sparse matrix to dense matrix
    X_train_transformed = pd.DataFrame(A_train.todense(),
                                       columns=vectorizer.get_feature_names_out())
    X_test_transformed = pd.DataFrame(A_test.todense(),
                                      columns=vectorizer.get_feature_names_out())
    oot_transformed = pd.DataFrame(A_oot.todense(),
                                   columns=vectorizer.get_feature_names_out())
    # create out-of-time series of labels
    y_oot = df_oot['answer']
    # TODO - needs work until here.
    # #### Load Champion Model and create copy as Challenger Model

    # In[16]:

    # Load champion model and create challenger model from deep-copy of champion_model.
    champion_model = load_champion_model()
    challenger_model = pickle.loads(pickle.dumps(champion_model))

    # #### Load hyperparameter values from config file
    # parse config file for the hyperparameter value ranges
    params = read_hyperparameters()

    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_params = select_best_params(X_train_transformed, y_train,
                                     challenger_model=challenger_model, params=params,
                                     stratified_kfold=stratified_kfold)
    challenger_model.set_params(**best_params)

    # #### Score calls using OOT
    y_pred_champion = champion_model.predict(oot_transformed)
    y_pred_challenger = challenger_model.predict(oot_transformed)

    # #### Calculate the Balanced Accuracy Score using OOT

    acc_champion = balanced_accuracy_score(y_oot, y_pred_champion)
    acc_challenger = balanced_accuracy_score(y_oot, y_pred_challenger)

    winner_flag = winner(acc_champion, acc_challenger)

    config_json_str = config_to_json()

    challenger_pickle = pickle.dumps(challenger_model)
    champion_pickle = pickle.dumps(champion_model)

    insert_magneto_summary_table(winner_flag=winner_flag,
                                 config_json_str=config_json_str,
                                 challenger_pickle=challenger_pickle,
                                 champion_pickle=champion_pickle)
