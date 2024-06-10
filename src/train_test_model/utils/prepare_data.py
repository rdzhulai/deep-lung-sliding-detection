import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

def prepare_data(df, sampled_size, train_test_size, train_val_size, seed, class_ratio=None):

    # Function to split dataset into train, val, and test sets
    def _split_dataset(df, train_test_size, val_test_size):
        train_df, test_val_df = train_test_split(df, test_size=train_test_size, stratify=df['video_class'])
        val_df, test_df = (None, test_val_df) if val_test_size == 1 else train_test_split(test_val_df, test_size=val_test_size, stratify=test_val_df['video_class'])
        return train_df, val_df, test_df

    # Function to balance dataset
    def _balance_dataset(df, target_ratio):
        sampled_df = pd.concat([
            df[df['video_class'] == 0].sample(n=int(df['video_class'].value_counts().get(1, 0) * target_ratio), replace=False),
            df[df['video_class'] == 1]
        ]).sample(frac=1)
        return sampled_df

    # Function to print class distribution
    def _print_class_distribution(sets):
        print("Class distribution: ", end="")
        for name, df in sets:
            if df is None:
                continue
            class_counts = df['video_class'].value_counts()
            print(f"{name}: {class_counts.get(1, 0)} positive, {class_counts.get(0, 0)} negative; ", end="")
        print("Sum:", sum(len(df) for _, df in sets if df is not None), end=" ")

    # Function to balance datasets
    def _balance_datasets(sets, target_ratio):
        class_ratios = [len(df[df['video_class'] == 0]) / len(df[df['video_class'] == 1]) for _, df in sets if df is not None]
        print("Class ratios: ", class_ratios)
        if not target_ratio:
            target_ratio = min(class_ratios)
        sets = [(name, _balance_dataset(df, target_ratio) if df is not None else None) for name, df in sets]
        class_ratios = [len(df[df['video_class'] == 0]) / len(df[df['video_class'] == 1]) for _, df in sets if df is not None]
        print("Class ratios: ", class_ratios, end=" ")
        return sets, target_ratio

    # Function to calculate ratios
    def _calculate_ratios(sets):
        total_size = sum(len(df) for _, df in sets if df is not None)
        output_string = " / ".join(
            [f"{len(df)/total_size}({df['video_class'].value_counts().get(0,0)}/{df['video_class'].value_counts().get(1,0)})" if df is not None else f"0(0/0)" for _, df in sets])
        print("Data split ratios:", output_string)
    
    np.random.seed(seed)
    
    # Map class labels to integers
    df['video_class'] = df['video_class'].map({'ls_p': 0, 'ls_a': 1})
    
    sampled_df = df if sampled_size == 0 else train_test_split(df, test_size=sampled_size, stratify=df['video_class'])[0]
    

    # Split dataset and filter data based on train, val, and test sets
    sets = _split_dataset(sampled_df[['video_name', 'video_class']].drop_duplicates(), train_test_size, train_val_size)

    sets = [(name, sampled_df[sampled_df['video_name'].isin(df['video_name'])] if df is not None else None) for name, df in zip(["Train", "Validation", "Test"], sets)]

    # Print class distribution before balancing
    _print_class_distribution(sets)

    # Balancing datasets
    sets, class_ratio = _balance_datasets(sets, class_ratio)

    # Print class distribution after balancing
    _print_class_distribution(sets)
    print()

    # Calculate ratios
    _calculate_ratios(sets)

    return [df for _, df in sets] + [class_ratio]
