import json
import argparse
from scipy.integrate import trapz
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker

def solution_quality_comparison(combined_data, dataset_name):
    max_sol = combined_data \
        .groupby('file_name')['status'] \
        .agg(lambda x: max(y['final_objective'] for y in x)) \
        .reset_index(name='max_finish_time')
    
    combined_data = pd.merge(combined_data, max_sol, on=['file_name'])
    combined_data['Z_best'] = combined_data['status'].apply(lambda x: x['final_objective'])
    combined_data['rescaled_sol'] = (combined_data['Z_best'] / combined_data['max_finish_time']) * 100

    # Aggregate the data to get mean avg_Z_best for each group
    aggregated_data = combined_data.groupby(['jobs', 'secondary_resources', 'source'])['rescaled_sol'].mean().rename('avg_rescaled_sol').reset_index()

    # Get unique values of secondary resources
    unique_m_values = sorted(aggregated_data['secondary_resources'].unique())

    # Increase font size of labels
    sns.set_context("paper", font_scale=1.5)

    # Create a single figure with multiple subplots
  
    fig, axes = plt.subplots(1, len(unique_m_values), figsize=(6 * len(unique_m_values), 3), sharex=True)

    # If there's only one subplot, axes is not a list, so we need to wrap it
    if len(unique_m_values) == 1:
        axes = [axes]

    bar_width = 0.2
    labels = aggregated_data['source'].unique()

    # Plot histogram for each m in a different subplot
    for ax, m_value in zip(axes, unique_m_values):
        group_data = aggregated_data[aggregated_data['secondary_resources'] == m_value]
        n_values = sorted(group_data['jobs'].unique())
        indices = range(len(n_values))

        for i, label in enumerate(labels):
            file_data = group_data[group_data['source'] == label]
            ax.bar(
                [index + i * bar_width for index in indices],
                file_data['avg_rescaled_sol'],
                width=bar_width,
                label=label 
            )

        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title(fr'Instance set {dataset_name}, $m = {m_value}$')
        ax.set_xlabel('n')
        ax.set_ylabel(r'$\overline{\text{% of highest } Z_{\text{best}}}$')
        ax.set_xticks([index + bar_width for index in indices])
        ax.set_ylim(bottom=70)
        ax.set_xticklabels(n_values)
        ax.grid(True)
        ax.set_ylim(top = 100)

    # Adjust layout
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    # Save the entire figure
    output_file_path = os.path.join(os.path.dirname(__file__), dataset_name + '_histogram_all_m.png')
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()

def analyze_convergence_rate(combined_data, dataset_name, time_limit):

    max_times = combined_data \
        .groupby('file_name')['status'] \
        .agg(lambda x: max(y['final_time'] for y in x)) \
        .reset_index(name='max_finish_time')
    
    combined_data = pd.merge(combined_data, max_times, on=['file_name'])

    for index, row in combined_data.iterrows():
        objective_values = [0]  
        times = [0]
    
        for stat in row['stats']:
            objective_values.append(int(stat['objective']))
            times.append(int(stat['time_spent_in_solver_in_milliseconds']))
            
        # Add final values for trapz calculation
        objective_values.append(objective_values[-1])
        times.append(row['max_finish_time']) 

        # Calculate area under the curve using trapz
        auc_value = trapz(objective_values, x=times)
        
        # Assign the calculated AUC value to 'area_under_curve'
        combined_data.at[index, 'area_under_curve'] = auc_value

    # Group by 'file_name' to find max AUC within each group
    max_auc = combined_data.groupby('file_name')['area_under_curve'].max().rename('max_area_under_curve').reset_index()
    # for file, item in combined_data.groupby(['file_name']):
    #     print(item)
    # Merge max_auc back into combined_data
    temp = pd.merge(combined_data, max_auc, on=['file_name'])
    # Calculate rescaled AUC
    temp['rescaled_auc'] = (temp['area_under_curve'] / temp['max_area_under_curve']) * 100

    # Group by 'jobs', 'secondary_resources', and 'source' and calculate mean of 'rescaled_auc'
    merged_data = temp.groupby(['jobs', 'secondary_resources', 'source']).agg({
        'rescaled_auc': 'mean',
    }).reset_index()
    
    # Get unique values of secondary resources
    unique_m_values = sorted(merged_data['secondary_resources'].unique())

    # Increase font size of labels
    sns.set_context("paper", font_scale=1.5)

    # Create a single figure with multiple subplots

    fig, axes = plt.subplots(1, len(unique_m_values), figsize=(6 * len(unique_m_values), 3), sharex=True)

    # If there's only one subplot, axes is not a list, so we need to wrap it
    if len(unique_m_values) == 1:
        axes = [axes]

    bar_width = 0.2
    labels = sorted(merged_data['source'].unique())

    # Plot histogram for each m in a different subplot
    for ax, m_value in zip(axes, unique_m_values):
        group_data = merged_data[merged_data['secondary_resources'] == m_value]
        n_values = sorted(group_data['jobs'].unique())
        indices = range(len(n_values))

        for i, label in enumerate(labels):
            file_data = group_data[group_data['source'] == label]
            ax.bar(
                [index + i * bar_width for index in indices],
                file_data['rescaled_auc'],
                width=bar_width,
                label=label 
            )

        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title(fr'Instance set {dataset_name}, $m = {m_value}$')
        ax.set_xlabel('n')
        ax.set_ylabel(r'$\overline{\text{% of highest AOC}}$')
        ax.set_xticks([index + bar_width for index in indices])
        ax.set_xticklabels(n_values)
        ax.set_ylim(bottom=70)
        ax.grid(True)
        ax.set_ylim(top = 100)


    # Adjust layout
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    # Save the entire figure
    output_file_path = os.path.join(os.path.dirname(__file__), dataset_name + '_histogram_all_m_auc.png')
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()


def analyze_number_of_conflicts(combined_data, dataset_name):
    combined_data['mean_conflicts'] = combined_data['stats'].apply(lambda stats: np.mean([stat['number_of_conflicts'] for stat in stats]))

    df_grouped = combined_data.groupby(['jobs', 'secondary_resources', 'source'], as_index=False)['mean_conflicts'].mean()
    max_conflicts = df_grouped['mean_conflicts'].max()

    unique_m = sorted(df_grouped['secondary_resources'].unique())
    if len(unique_m) == 1:
        axes = [axes]
    n_values = sorted(df_grouped['jobs'].unique())
    indices = range(len(n_values))

    # Create a single figure with multiple subplots
    sns.set_context("paper", font_scale=1.5)

    fig, axes = plt.subplots(1, len(unique_m), figsize=(6 * len(unique_m), 3), sharex=True)

    bar_width = 0.2
    sources = sorted(combined_data['source'].unique())

    # Plot histogram for each m in a different subplot
    for ax, m_value in zip(axes, unique_m):
        group_data = df_grouped[df_grouped['secondary_resources'] == m_value]
        
        for i, source in enumerate(sources):
            file_data = group_data[group_data['source'] == source]

            ax.bar(
                [index + i * bar_width for index in indices],
                file_data['mean_conflicts'],
                width=bar_width,
                label=source 
            )

        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_title(fr'Instance set {dataset_name}, $m = {m_value}$')
        ax.set_xlabel('n')
        ax.set_ylabel(r'$\overline{K}$ (000s)')
        ax.set_xticks([index + bar_width for index in indices])
        ax.set_xticklabels(n_values)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '%.0f' % (y * 1e-3)))
        ax.grid(True)
        ax.set_ylim(top = max_conflicts + 10000)

    # Adjust layout
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    # # Save the entire figure
    output_file_path = os.path.join(os.path.dirname(__file__), dataset_name + '_histogram_all_m_conflicts.png')
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()

def status_analysis(combined_data, dataset_name):

    for index, row in combined_data.iterrows():
        if row['status']['final_objective'] == 0:
            row['status']['status'] = 'unknown'
    
    combined_data['optimal'] = combined_data['status'].apply(lambda x: 1 if x['status'] == 'optimal' else 0)
    combined_data['satisfiable'] = combined_data['status'].apply(lambda x: 1 if x['status'] == 'satisfiable' else 0)
    combined_data['unknown'] = combined_data['status'].apply(lambda x: 1 if x['status'] == 'unknown' else 0)
    combined_data['infeasible'] = combined_data['status'].apply(lambda x: 1 if x['status'] == 'infeasible' else 0)

    # Grouping by 'jobs', 'secondary_resources', 'source' and summing up the counts
    result = combined_data.groupby(['jobs', 'secondary_resources', 'source']).agg({
        'optimal': 'sum',
        'satisfiable': 'sum',
        'unknown': 'sum',
        'infeasible': 'sum'
    }).reset_index()

    result_sorted = result.sort_values(by=['source', 'secondary_resources', 'jobs'])

    result_sorted.to_csv(f'{dataset_name}_batch_status', index=False)

def read_json_file(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and process JSON output file.")
    parser.add_argument("--json_file1", type=str, required=True, help="Path to JSON output file 1.")
    parser.add_argument("--json_file2", type=str, required=True, help="Path to JSON output file 1.")
    parser.add_argument("--json_file3", type=str, required=True, help="Path to JSON output file 1.")
    parser.add_argument("--source_name1", type=str, required=True, help="Label used for plots.")
    parser.add_argument("--source_name2", type=str, required=True, help="Label used for plots.")
    parser.add_argument("--source_name3", type=str, required=True, help="Label used for plots.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset.")
    parser.add_argument("--experiment_timelimit", type=str, required=True, help="Time limit of the experiment (per run).")
    '''
    python process_gourd_results.py --json_file1 experiment-balanced-vsids.json --json_file2 experiment-balanced-hdf.json --json_file3 experiment-balanced-vsidsdensity.json --source_name1 VSIDS --source_name2 HDF --source_name3 VSIDS+Density --dataset_name B --experiment_timelimit 900000
    '''
    args = parser.parse_args()

    json_file1 = args.json_file1.strip()
    name1 = args.source_name1.strip()
    json1 = read_json_file(json_file1)
    df1 = pd.DataFrame(json1)
    df1['source'] = name1

    json_file2 = args.json_file2.strip()
    name2 = args.source_name2.strip()
    json2 = read_json_file(json_file2)
    df2 = pd.DataFrame(json2)
    df2['source'] = name2

    json_file3 = args.json_file3.strip()
    name3 = args.source_name3.strip()
    json3 = read_json_file(json_file3)
    df3 = pd.DataFrame(json3)
    df3['source'] = name3

    combined_data = pd.concat([df1, df2, df3])
    dataset_name = args.dataset_name.strip()
    time_limit = int(args.experiment_timelimit.strip())

    # analyze_convergence_rate(combined_data.copy(), dataset_name, time_limit)
    status_analysis(combined_data.copy(), dataset_name)
    # analyze_number_of_conflicts(combined_data.copy(), dataset_name)
    

    only_solved = combined_data[~combined_data['status'].apply(lambda x: x['status']).isin(['unknown', 'infeasible'])]
    # solution_quality_comparison(only_solved, dataset_name)
    
# python process_gourd_results.py --json_file1 experiment_skewed/results/experiment-skewed-vsids.json --json_file2 experiment_skewed/results/experiment-skewed-vsidsdensity.json --json_file3 experiment_skewed/results/experiment-skewed-hdf.json --source_name1 VSIDS --source_name2 VSIDS+Density --source_name3 HDF --dataset_name S --experiment_timelimit 900000 
# python process_gourd_results.py --json_file1 experiment_balanced/results/experiment-balanced-vsids100max.json --json_file2 experiment_balanced/results/experiment-balanced-vsidsdensity.json --json_file3 experiment_balanced/results/experiment-balanced-hdf.json --source_name1 VSIDS --source_name2 VSIDS+Density --source_name3 HDF --dataset_name B --experiment_timelimit 900000 