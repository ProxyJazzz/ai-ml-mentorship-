from pathlib import Path

from src.load_data import load_student_data
from src.analysis import calculate_statistics, add_result_column
from src.visualization import plot_subject_histogram, plot_average_bar_chart


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "students.csv"


def main():
    df = load_student_data(DATA_PATH)
    if df is None:
        return

    df = add_result_column(df)

    stats = calculate_statistics(df)

    print("\n📊 SUBJECT STATISTICS")
    for subject, values in stats.items():
        print(subject, values)

    print("\n📋 FINAL DATA")
    print(df)

    plot_subject_histogram(df)
    plot_average_bar_chart(df)


if __name__ == "__main__":
    main()
