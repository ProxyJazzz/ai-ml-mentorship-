import matplotlib.pyplot as plt

def plot_subject_histogram(df):
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        plt.figure()
        df[col].plot(kind="hist", bins=10, title=f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()


def plot_average_bar_chart(df):
    plt.figure()
    plt.bar(df["name"], df["average"])
    plt.title("Average Marks per Student")
    plt.xlabel("Student")
    plt.ylabel("Average Marks")
    plt.xticks(rotation=45)
    plt.show()
