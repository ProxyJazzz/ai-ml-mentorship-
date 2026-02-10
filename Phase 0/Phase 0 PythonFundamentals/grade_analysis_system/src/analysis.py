def calculate_statistics(df):
    stats = {}

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        stats[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std()
        }

    return stats


def add_result_column(df, pass_marks=50):
    numeric_cols = df.select_dtypes(include="number").columns

    df["average"] = df[numeric_cols].mean(axis=1)
    df["result"] = df["average"].apply(
        lambda x: "Pass" if x >= pass_marks else "Fail"
    )

    return df
