import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_users_time(df):
    df.columns = ['ID', 'first & last name', 'first time', 'second time', 'third time', 'irrelevant1', 'irrelevant2',
                  'irrelevant3']
    df = df.drop(0).reset_index(drop=True)
    df = df[['ID', 'first & last name', 'first time', 'second time', 'third time']]
    df[['first time', 'second time', 'third time']] = df[['first time', 'second time', 'third time']].astype(float)
    df['first & last name'] = df['first & last name'].apply(lambda x: x[::-1])

    # Plot the data
    plt.figure(figsize=(12, 16))
    for i in range(len(df)):
        plt.plot(['First Time', 'Second Time', 'Third Time'],
                 [df.loc[i, 'first time'], df.loc[i, 'second time'], df.loc[i, 'third time']], marker='o',
                 label=df.loc[i, 'first & last name'])

    plt.xlabel('Usage')
    plt.ylabel('Time to Complete Task')
    plt.title('Time to Complete Task Over Multiple Uses of the System')
    plt.legend(title='Users', bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid(True)
    plt.show()


def plot_avarage_time_per_letter(df):
    df.columns = ['ID', 'first & last name', 'first time', 'second time', 'third time', 'irrelevant1', 'irrelevant2',
                  'irrelevant3']
    df = df.drop(0).reset_index(drop=True)
    df = df[['ID', 'first & last name', 'first time', 'second time', 'third time']]
    df[['first time', 'second time', 'third time']] = df[['first time', 'second time', 'third time']].astype(float)

    # Reverse the Hebrew usernames
    df['first & last name'] = df['first & last name'].apply(lambda x: x[::-1])

    # Calculate average time per letter for the first and second times
    df['name_length'] = df['first & last name'].apply(len)
    df['avg_time_first'] = df['first time'] / df['name_length']
    df['avg_time_second'] = df['second time'] / df['name_length']

    # Calculate the overall average time per letter
    avg_time_first = df['avg_time_first'].mean()
    avg_time_second = df['avg_time_second'].mean()

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.bar(['First Time', 'Second Time'], [avg_time_first, avg_time_second], color=['blue', 'orange'])

    plt.xlabel('Usage')
    plt.ylabel('Average Time per Letter (seconds)')
    plt.title('Average Time per Letter to Write Names Using the System')
    plt.grid(axis='y')

    plt.show()


def avarage_time_per_letter_by_user(df):
    df.columns = ['ID', 'first & last name', 'first time', 'second time', 'third time', 'irrelevant1', 'irrelevant2',
                  'irrelevant3']
    df = df.drop(0).reset_index(drop=True)
    df = df[['ID', 'first & last name', 'first time', 'second time', 'third time']]
    df[['first time', 'second time', 'third time']] = df[['first time', 'second time', 'third time']].astype(float)

    # Reverse the Hebrew usernames
    df['first & last name'] = df['first & last name'].apply(lambda x: x[::-1])

    # Calculate average time per letter for the first and second times
    df['name_length'] = df['first & last name'].apply(len)
    df['avg_time_first'] = df['first time'] / df['name_length']
    df['avg_time_second'] = df['second time'] / df['name_length']

    # Paired bar chart
    x = np.arange(len(df['first & last name']))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width / 2, df['avg_time_first'], width, label='First Time')
    bars2 = ax.bar(x + width / 2, df['avg_time_second'], width, label='Second Time')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Users',fontsize=20)
    ax.set_ylabel('Minutes (per letter)', fontsize=20)
    ax.set_title('Average Time per Letter')
    ax.set_xticks(x)
    ax.set_xticklabels(df['first & last name'], rotation=90,fontsize=16)
    ax.legend()

    fig.tight_layout()
    plt.show()


def load_df(path):
    return pd.read_excel(path, sheet_name='גיליון1', header=1)


import seaborn as sns

def violin_plot_time_per_letter(df):
    df.columns = ['ID', 'first & last name', 'first time', 'second time', 'third time', 'irrelevant1', 'irrelevant2', 'irrelevant3']
    df = df.drop(0).reset_index(drop=True)
    df = df[['ID', 'first & last name', 'first time', 'second time', 'third time']]
    df[['first time', 'second time', 'third time']] = df[['first time', 'second time', 'third time']].astype(float)

    # Reverse the Hebrew usernames
    df['first & last name'] = df['first & last name'].apply(lambda x: x[::-1])

    # Calculate average time per letter for the first and second times
    df['name_length'] = df['first & last name'].apply(len)
    df['avg_time_first'] = df['first time'] / df['name_length']
    df['avg_time_second'] = df['second time'] / df['name_length']

    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Time per Letter (minutes)': pd.concat([df['avg_time_first'], df['avg_time_second']]),
        'Attempt': ['First Time'] * len(df) + ['Second Time'] * len(df)
    })

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Attempt', y='Time per Letter (minutes)', data=plot_data)
    plt.title('Distribution of Average Time per Letter')
    plt.grid(True)
    plt.show()

# Example usage with the DataFrame
file_path = r'C:\Users\40gil\Desktop\final_project\tensor_training\PostModelAnalyze\vals\val_users_times.xlsx'
df = pd.read_excel(file_path, sheet_name='גיליון1', header=1)
violin_plot_time_per_letter(df)


if __name__ == '__main__':
    df = load_df(r'C:\Users\40gil\Desktop\final_project\tensor_training\PostModelAnalyze\vals\val_users_times.xlsx')

    # plot_users_time(df)
    # plot_avarage_time_per_letter(df)
    violin_plot_time_per_letter(df)


    #C:\Users\40gil\Desktop\final_project\tensor_training\PostModelAnalyze\plots
