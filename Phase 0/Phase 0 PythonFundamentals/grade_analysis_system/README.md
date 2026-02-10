# 📊 Grade Analysis System (Phase 0 – Python Fundamentals)

A beginner-friendly **Grade Analysis System** built using **Python, Pandas, and Matplotlib**.  
This project is part of **Phase 0 (Foundation Building)** in a Machine Learning learning path and focuses on **data handling, statistics, and visualization**.

---

## 🎯 Objective

To build a complete Python program that:
- Reads student marks from a CSV file
- Performs statistical analysis (mean, median, standard deviation)
- Calculates student averages and pass/fail results
- Visualizes data using graphs

This project strengthens **core Python + data analysis foundations**, which are essential before moving to Machine Learning.

---

## 🧠 Concepts Covered

- Python functions & modular programming
- File handling using CSV
- Pandas DataFrame operations
- Handling numeric vs non-numeric data safely
- Basic statistics:
  - Mean
  - Median
  - Standard Deviation
- Data visualization using Matplotlib
- Clean project structure (industry-style)

---

## 📁 Project Structure

grade_analysis/
│
├── main.py # Entry point of the program
├── students.csv # Input dataset
│
└── src/
├── init.py
├── load_data.py # CSV loading logic
├── analysis.py # Statistics & result calculation
└── visualization.py # Graph plotting


---

## 📄 Dataset Format (`students.csv`)

```csv
name,maths,physics,chemistry
Ankit,85,90,88
Rahul,72,65,70
Priya,95,92,94
Sneha,60,58,62
Aman,45,50,48


▶️ How to Run the Project
1️⃣ Go to project folder
cd Phase0PythonFundamentals/grade_analysis

2️⃣ Install dependencies (one-time)
python -m pip install pandas matplotlib

3️⃣ Run the program
python main.py

✅ Output
🖥 Terminal Output

Subject-wise statistics (mean, median, std)

Final table with:

Average marks

Pass / Fail result

📊 Visual Output

Histograms for each subject

Bar chart showing average marks per student

(Graphs open in separate windows.)

🛠 Key Implementation Details

Only numeric columns are used for calculations
(avoids common Pandas errors with string data)

Modular design:

Easy to debug

Easy to extend

No hard-coded column indices (robust to CSV changes)

🚀 Future Improvements

Export results to CSV or text report

Add percentiles and boxplots

Add data validation & error handling

Extend to Phase-1 (Exploratory Data Analysis)

Convert into Machine Learning pipeline

📌 Learning Outcome

By completing this project, you gain:

Confidence in Python basics

Strong understanding of Pandas workflows

Real-world debugging experience

Foundation required for Machine Learning

👤 Author

Ankit
Engineering Student | AIML Learner
