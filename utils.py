import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Let's first define a function that creates heatmaps for frequency tables
# as several plots in the EDA from now on will look like this
custom_cmap = sns.light_palette("#4D4D4D", as_cmap=True)

def plot_heatmap_with_percentages(
    df: pd.DataFrame,
    col: str,
    xlabel: str,
    ylabel: str,
    title: str = None,
    condition_col: str = None,
    count_col: Optional[str] = 'call_or_bsr_in_next_30_days',
    define_new_col: bool = True,
    custom_cmap = custom_cmap,
    save_file_name: str = None
) -> None:
    """
    Plots a heatmap with counts and percentages for a given condition.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing the data.
    - col: str - The column name to create the new condition column.
    - condition_col: str - The column name for the condition to be checked.
    - title: str - The title of the plot.
    - xlabel: str - The label for the x-axis.
    - ylabel: str - The label for the y-axis.
    - condition: Optional[str] - The column name to apply the condition on. Default is 'ban_dsc_sum_amt_0'.
    """

    
    if define_new_col:
        df[col] = df[condition_col] < 0

    freq_tab = pd.crosstab(df[col], df[count_col], rownames=[col], colnames=[count_col])

    total_sum = freq_tab.sum().sum()
    percentage_table = freq_tab.div(total_sum) * 100

    combined_table = freq_tab.astype(str) + '\n(' + percentage_table.round(1).astype(str) + '%)'

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(freq_tab, annot=combined_table, fmt='', cmap=custom_cmap, cbar=True, linewidths=.5)
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_file_name is not None:
        plt.savefig(save_file_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_horizontal_conditional_histogram(df: pd.DataFrame,
                                     group_by_column: str,
                                     count_column: str = 'call_or_bsr_in_next_30_days',
                                     ylabel: str = None,
                                     xlabel: str = None,
                                     title: str = None,
                                     figsize: tuple = (15, 10),
                                     xticks: list = None,
                                     yticks: list = None,
                                     save_file_name: str = None):

    grouped_counts = df.groupby([group_by_column, count_column]).size().unstack(fill_value=0)

    grouped_counts.plot(kind='barh', stacked=False, figsize=figsize) # ['#E0E0E0', '#A0A0A0'])

    for i, (_, row) in enumerate(grouped_counts.iterrows()):
        group_count = row.sum()
        for j, (_, count) in enumerate(row.items()):
            percentage = (count / group_count) * 100
            plt.text(count, 
                     i + j * 0.25 - 0.1,  # Adjust the position to avoid overlap
                     f'{count:,}\n({percentage:.1f}%)',
                     ha='left', 
                     va='center',
                     color='#404040',
                     fontsize=10)

    # Customize the plot
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    if xticks:
        plt.xticks(ticks=range(len(xticks)), labels=xticks)
    if yticks:
        plt.yticks(ticks=yticks)

    legend_labels = grouped_counts.columns.tolist()
    plt.legend(legend_labels, title=count_column, loc='best')
    
    # Save the plot if a save file path is passed
    if save_file_name is not None:
        plt.savefig(save_file_name)
        
    plt.tight_layout()
    plt.show()
    
def plot_vertical_conditional_histogram(df: pd.DataFrame,
                                        group_by_column: str,
                                        count_column: str = 'call_or_bsr_in_next_30_days',
                                        ylabel: str = None,
                                        xlabel: str = None,
                                        title: str = None,
                                        figsize: tuple = (15, 10),
                                        xticks: list = None,
                                        yticks: list = None,
                                        save_file_name: str = None):

    grouped_counts = df.groupby([group_by_column, count_column]).size().unstack(fill_value=0)

    # Plot vertical bars
    grouped_counts.plot(kind='bar', stacked=False, figsize=figsize, color=['#95d0fc', '#0343df'])

    total_counts = len(df)
    for i, (category, row) in enumerate(grouped_counts.iterrows()):
        group_count = row.sum()
        for j, (value, count) in enumerate(row.items()):
            percentage = (count / group_count) * 100
            plt.text(i, 
                     count + 0.05 * group_count,  # Adjust the position to avoid overlap
                     f'{count:,}\n({percentage:.1f}%)',
                     ha='center', 
                     va='bottom',
                     color='#404040',
                     fontsize=10)

    # Customize the plot
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    if xticks:
        plt.xticks(ticks=range(len(xticks)), labels=xticks)
    if yticks:
        plt.yticks(ticks=yticks)
    
    # Save the plot if a save file path is passed
    if save_file_name is not None:
        plt.savefig(save_file_name)
        
    plt.tight_layout()
    plt.show()

    
def plot_column_histogram(df: pd.DataFrame, column: str, ylabel: str = None, xlabel: str = None, title: str = None, figsize: tuple =(15, 10)):
    # Count the True values for each column
    counts = df[column].value_counts()
    # Create a bar plot
    plt.figure(figsize=figsize)
    bars = plt.barh(counts.index, counts.values, color='#E0E0E0')

    # Add percentage labels
    total_counts = len(df)
    for bar in bars:
        
        percentage = (bar.get_width() / total_counts) * 100
        count = int(bar.get_width())
        plt.text(bar.get_width(), 
                bar.get_y() + bar.get_height()/2., 
                f'{count:,}\n({percentage:.1f}%)',
                ha='left', 
                va='center',
                color='#404040',
                fontsize=10)

    # Customize the plot
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()
    
def plot_histogram(df: pd.DataFrame, column: str, bins: int = 100, title: str = None, xlabel: str = None, ylabel: str = None, figsize: tuple = (10, 6)):
    """
    Plots a histogram of a specified column in a DataFrame.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing the data.
    - column_name: str - The name of the column to plot.
    - bins: int - The number of bins for the histogram (default is 10).
    - title: str - The title of the plot (optional).
    - xlabel: str - The label for the x-axis (optional).
    - ylabel: str - The label for the y-axis (optional).
    - figsize: tuple - The size of the figure (default is (10, 6)).
    """
    # Use logarithmic bins
    min_val, max_val = df[column].min(), df[column].max()
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)

    # Plotting the histogram
    plt.figure(figsize=figsize)
    plt.hist(df[column], bins=log_bins, edgecolor='black')
    plt.xscale('log')

    # Adding titles and labels if provided
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str, xlabel: str, ylabel: str, title: str = None, figsize: Tuple[int, int] = (10,6)) -> None:
    """
    Plots a scatter plot of two integer variables from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x_col (str): The name of the column to be used for the x-axis.
    - y_col (str): The name of the column to be used for the y-axis.
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - title (str): The title of the plot.
    - figsize (Tuple[int, int]): The size of the figure (width, height).

    Returns:
    - None
    """
    plt.figure(figsize=figsize)
    plt.scatter(df[xcol], df[ycol])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.grid(True)
    plt.show()
    
def plot_percentages_line_plot(df: pd.DataFrame,
                               group_by_column: str,
                               count_column: str = 'call_or_bsr_in_next_30_days',
                               ylabel: str = None,
                               xlabel: str = None,
                               title: str = None,
                               figsize: tuple = (15, 10),
                               xticks: list = None,
                               yticks: list = None,
                               save_file_name: str = None):

    # Calculate total counts for each group
    total_counts = df.groupby(group_by_column).size()

    # Filter the dataframe for the case when count_column == 1
    filtered_df = df[df[count_column] == 2]

    # Calculate counts for the filtered DataFrame
    filtered_counts = filtered_df.groupby(group_by_column).size()

    # Calculate percentages over the entire DataFrame
    percentages = (filtered_counts / total_counts) * 100
    
    # Plot the line plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(percentages.index, percentages, marker='o', label=f'{count_column} == 1')

    # Add text annotations for percentages and counts
    for group in percentages.index:
        percentage = percentages.loc[group]
        actual_count = filtered_counts.loc[group]
        ax.text(group, percentage, f'{percentage:.1f}%\n({actual_count})', ha='center', va='bottom', fontsize=10)

    # Customize the plot
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xticks:
        ax.set_xticks(ticks=range(len(xticks)))
        ax.set_xticklabels(labels=xticks)
    if yticks:
        ax.set_yticks(ticks=yticks)

    # Add legend
    ax.legend(title=count_column)
    
    # Save the plot if a save file path is passed
    if save_file_name is not None:
        plt.savefig(save_file_name)
        
    plt.tight_layout()
    plt.show()

def plot_percentages_line_plot_all_groups(df: pd.DataFrame,
                               group_by_column: str,
                               count_column: str = 'call_or_bsr_in_next_30_days',
                               ylabel: str = None,
                               xlabel: str = None,
                               title: str = None,
                               figsize: tuple = (15, 10),
                               xticks: list = None,
                               yticks: list = None,
                               save_file_name: str = None,
                               order: list = None):

    # Calculate total counts for each group
    total_counts = df.groupby(group_by_column).size()

    # Get unique values in the count_column
    unique_values = df[count_column].unique()

    # Plot the line plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Groups 
    groups = df[group_by_column].unique()

    for value in unique_values:
        # Filter the dataframe for the current value in count_column
        filtered_df = df[df[count_column] == value]

        # Calculate counts for the filtered DataFrame
        filtered_counts = filtered_df.groupby(group_by_column).size()

        for group in groups:
            if group not in filtered_counts.index:
                filtered_counts[group] = 0
                
        # Calculate percentages over the entire DataFrame
        percentages = (filtered_counts / total_counts) * 100

        # Sort percentages according to the specified order
        if order:
            percentages = percentages.reindex(order)
        
        # Plot the line for the current value
        ax.plot(percentages.index, percentages, marker='o', label=f'{count_column} == {value}')

        # Add text annotations for percentages and counts
        for group in percentages.index:
            if pd.notna(percentages.loc[group]):
                percentage = percentages.loc[group]
                actual_count = filtered_counts.loc[group]
                ax.text(group, percentage, f'{percentage:.1f}%\n({actual_count})', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(group, 0, f'0.0%\n(0)', ha='center', va='bottom', fontsize=10)

    # Customize the plot
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xticks:
        ax.set_xticks(ticks=range(len(xticks)))
        ax.set_xticklabels(labels=xticks, rotation=90)
    if yticks:
        ax.set_yticks(ticks=yticks)

    # Add legend
    ax.legend(title=count_column, loc = 'center right')
    
    # Save the plot if a save file path is passed
    if save_file_name is not None:
        plt.savefig(save_file_name)
        
    plt.tight_layout()
    plt.show()