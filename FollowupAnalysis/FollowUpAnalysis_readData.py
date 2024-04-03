from __future__ import annotations
import pandas as pd
import json
import os

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def read_followup_json_files_to_df_byusmletest(path:str):
    '''
    input:
    - path: the path to the standard json file containing a list of standard followup question object specified in the README.md

    output:
    - df: the dataframe with each row representing a usmle test and some properties extracted from the results
        - header of the dataframe: ["usmle_test",'follow_up_q',"number_of_question",
                                    "basic_knowledge_count_of_false",
                                'interpretation_and_association_count_of_false',
                                'total_count_of_false']

    '''
    json_list = json.load(open(path))
    df = pd.DataFrame(columns=["usmle_test",'follow_up_q',"number_of_question",
                                "basic_knowledge_count_of_false",
                               'interpretation_and_association_count_of_false',
                               'total_count_of_false'])
    for index, js in enumerate(json_list):
        try:
            num_of_question = len(js['follow_up_q'])
            question_for_basic_knowledge_count_false = len([x for x in js['follow_up_q'] if x['key'] == 'question_for_basic_knowledge' and x['annotate'] == False])
            question_for_interpretation_and_association_count_false = len([x for x in js['follow_up_q'] if x['key'] == 'question_for_interpretation_and_association' and x['annotate'] == False])
            total_count_of_false = len([x for x in js['follow_up_q'] if x['annotate'] == False])
        except Exception as e:
            print(e)
            print(js['index'])
            print(js['usmle_test'])
        df.loc[index] = [
                            js['usmle_test'],js['follow_up_q'], num_of_question,
                            question_for_basic_knowledge_count_false,
                            question_for_interpretation_and_association_count_false,
                            total_count_of_false]
            
    return df

def read_followup_json_files_to_df_byfollowupq(path:str):
    '''
    input:
    - path: the path to the standard json file containing a list of standard followup question object specified in the README.md

    output:
    - df: the dataframe with each row representing a followup question and its attributes
    '''
    json_list = json.load(open(path))
    flist = []
    for jsonobj in json_list:
        flist.extend(jsonobj['follow_up_q'])
    df = pd.DataFrame(flist)
    return df


def summarize_performance(orignialdf:pd.DataFrame, on:str):
    '''
    input:
    orignialdf: the original dataframe 
    on: the column name to group by

    output:
    df: the summarized dataframe
    
    '''
    if on not in orignialdf.columns:
        raise ValueError(f'{on} is not in the column of the dataframe')
    
    df = orignialdf.groupby(on).agg({'question':'count','annotate':'sum'}).reset_index()
    
    df['absolute_false'] = df['question'] - df['annotate']
    df['percentage_of_true'] =  df['annotate']/df['question']
    df['percentage_of_false'] = 1- df['annotate']/df['question']
    df.columns = [f'{on}_cat','count_of_all','count_of_true',
                  'count_of_false','percentage_of_true','percentage_of_false']
    if on == 'classification':
        medical_knowledge_categories = json.load(open(f'{script_dir}/medi_kno_cat.json'))
        df['category'] = df['classification_cat'].map(lambda x: [category for category, classifications in medical_knowledge_categories.items() if x in classifications][0])
    df.set_index(f'{on}_cat',inplace=True)

    return df



