import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df_tmp = pd.merge(df, ndc_df[['NDC_Code', 'Proprietary Name']], left_on='ndc_code', right_on='NDC_Code')
    df_tmp.rename(columns={'Proprietary Name':'generic_drug_name'},inplace=True)
    df_tmp = df_tmp.drop(['NDC_Code'], axis=1)

    return df_tmp

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''

    df_first_encounter=df.sort_values(["patient_nbr","encounter_id"]).groupby("patient_nbr").head(1)
    return df_first_encounter.reset_index(drop=True)



    return df


#Question 6
def patient_dataset_splitter(processed_df, key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    train_size = 0.6
    test_val_size = 0.2
    dfs = processed_df.iloc[np.random.permutation(len(processed_df))]
    unique_patient_id = dfs[key].unique()
    train_size = round(len(unique_patient_id) * train_size)
    test_val_size = round(len(unique_patient_id) * test_val_size)
    train_sample_patient_id = unique_patient_id[:train_size]
    train_df = dfs[dfs[key].isin(train_sample_patient_id)]
    test_sample_patient_id = unique_patient_id[train_size:train_size + test_val_size]
    test_df = dfs[dfs[key].isin(test_sample_patient_id)]
    val_sample_patient_id = unique_patient_id[train_size + test_val_size:]
    val_df= dfs[dfs[key].isin(val_sample_patient_id)]
    
    return train_df, val_df, test_df


#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_column)
        output_tf_list.append(tf_categorical_feature_column)
        
    return output_tf_list


#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    zscore_normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature =  tf.feature_column.numeric_column(key=col, default_value = default_value, 
                                                           normalizer_fn=zscore_normalizer, dtype=tf.float64)

    return tf_numeric_feature


#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_pred = df[col].apply(lambda x: 1 if x >=5 else 0)
    
    return student_binary_pred.to_numpy().flatten()
