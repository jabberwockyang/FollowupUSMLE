from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

## plot
def plot_hist_performance_bycat(df: pd.DataFrame ,col_of_group: str,col_of_count: str,ylim: int,xlim:int):
    '''
    input:
    - df: the dataframe loaded from the standard json file by the function read_followup_json_files_to_df_byusmletest
        - header of the dataframe: ["usmle_test",'follow_up_q',"number_of_question", "basic_knowledge_count_of_false",'interpretation_and_association_count_of_false', 'total_count_of_false']
    - col_of_group: the column name to group by
    - col_of_count: the column name to count
    - ylim: the y limit of the histogram
    - xlim: the x limit of the histogram

    output:
    - a histogram of the count of `col_of_count` for each category in `col_of_group`

    '''
    
    df=df.sort_values(by=col_of_group)
    categories = df[col_of_group].unique().tolist()
        
    catgory_count = len(categories)
    fig, axes = plt.subplots(1, catgory_count, figsize=((5*catgory_count), 5))  # Create a figure and a set of subplots
    
    subplot_index = 0
    
    for categ in categories:
        subset = df[df[col_of_group] == categ]
        # Plot on the appropriate subplot
        axes[subplot_index].hist(subset[col_of_count], 
                                bins=range(xlim),
                                #add border to histogram
                                edgecolor='white',  linewidth=1.5
                                )
        axes[subplot_index].set_title(f'Histogram of {col_of_count} \n for {categ} performance')
        
        axes[subplot_index].set_xlabel(col_of_count)
        axes[subplot_index].set_ylabel('count')
        axes[subplot_index].set_ylim([0, ylim]),
        axes[subplot_index].grid(False)
        
        # Move to the next subplot for the next performance
        subplot_index += 1

    # Display the plots
    plt.tight_layout()
    plt.show()



def plot_hist_for_followup_perform_byclass(originialdf: pd.DataFrame, col: str,colforcat: str, percentage:bool = False):
    '''
    input:
    - originialdf: the dataframe loaded from the standard json file by the function `read_followup_json_files_to_df_byfollowupq` and `summarize_performance`
        - header of the dataframe: [f'{on}_cat','count_of_all','count_of_true',
                    'count_of_false','percentage_of_false','percentage_of_true']
    - col: the column name to plot, can be absolute count or percentage if percentage, set the percentage to True
    - colforcat: the column name to assign color for each category
    - percentage: whether to plot the percentage of the count 
    
    output:
    - a histogram of the count of `col` for each category in `colforcat` with different color
    
    '''
    color_mapping = {
    'Basic Medicine': 'blue',
    'Clinical Medicine': 'green',
    'Others': 'red'}

    df = originialdf.copy()
    df['color'] = df[colforcat].map(color_mapping)
    
    df = df.sort_values(by=[col], ascending=False)
    df = df.drop('clinical')

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    # We use bar plot because histograms do not naturally support multiple categories
    plt.bar(df.index, df[col], color=df['color'])

    # Add titles and labels
    plt.title(f'Distribution of {col} ')
    plt.xlabel('Fields')
    plt.ylabel('Count/percentage')
    if percentage: 
        plt.ylim((0,1))
    plt.xticks(rotation=90)  # Rotate the x labels to show them more clearly

    # Add a legend
    handles = [plt.Rectangle((0,0),1,1, color=color_mapping[label]) for label in color_mapping]
    plt.legend(handles, color_mapping.keys())

    plt.tight_layout()
    plt.show()
