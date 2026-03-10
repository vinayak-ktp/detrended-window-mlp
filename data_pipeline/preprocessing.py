def split_timeseries(df, train_size=0.7, val_size=0.15):
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def normalize_features(train_df, val_df, test_df, to_normalize=None):
    stats = {}
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    if not to_normalize:
        to_normalize = train_df.columns

    for col in to_normalize:
        mean = train_df[col].mean()
        std = train_df[col].std() + 1e-8

        train_out[col] = (train_df[col] - mean) / std
        val_out[col] = (val_df[col]   - mean) / std
        test_out[col] = (test_df[col]  - mean) / std

        stats[col] = {'mean': mean, 'std': std}

    return train_out, val_out, test_out, stats
