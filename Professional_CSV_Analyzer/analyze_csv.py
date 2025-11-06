import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import matplotlib.animation as animation
import warnings
import sys
import os

warnings.filterwarnings("ignore")

def summarize_data(df):
    """
    Generate a concise textual summary of the dataset.
    """
    summary = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().sum(),
        "Duplicates": df.duplicated().sum(),
        "Numeric Columns": len(df.select_dtypes(include=np.number).columns),
        "Categorical Columns": len(df.select_dtypes(exclude=np.number).columns),
    }

    print("\n ----- DATA SUMMARY -----")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nTop 5 Rows:")
    print(df.head())

    print("\n🔹 Statistical Overview:")
    print(df.describe(include='all').transpose().head(10))
    return summary


def visualize_correlation(df):
    """
    Create and save a correlation heatmap.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()
        print(" Correlation heatmap saved as 'correlation_heatmap.png'")


def visualize_distribution(df):
    """
    Create and save distribution plots for all numeric features.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(figsize=(12, 10), bins=20, color='#1976D2', edgecolor='black')
        plt.suptitle('Feature Distributions', fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig("feature_distributions.png")
        plt.close()
        print(" Feature distributions saved as 'feature_distributions.png'")


def animated_trend(df):
    """
    Create an animated line plot showing trends in the first numeric column.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return

    col = numeric_cols[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r-', animated=True)

    def init():
        ax.set_xlim(0, len(df))
        ax.set_ylim(df[col].min(), df[col].max())
        ax.set_title(f"Animated Trend of '{col}'", fontsize=14, fontweight="bold")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(df[col].iloc[frame])
        ln.set_data(xdata, ydata)
        return ln,

    ani = animation.FuncAnimation(fig, update, frames=range(len(df)), init_func=init, blit=True, interval=30)
    ani.save("animated_trend.gif", writer="pillow")
    plt.close()
    print(" Animated trend saved as 'animated_trend.gif'")


def clean_data(df):
    """
    Handle missing values and duplicates.
    """
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna("Unknown")
    print("\n Data cleaned successfully.")
    return df


def generate_profile(df):
    """
    Create a full interactive HTML profiling report.
    """
    profile = ProfileReport(df, title=" Professional CSV Analysis Report", explorative=True)
    profile.to_file("analysis_report.html")
    print(" Full EDA report saved as 'analysis_report.html'")


def analyze_csv(csv_path):
    """
    Complete professional data analysis pipeline.
    """
    print(f"\n Loading dataset from: {csv_path}\n")
    df = pd.read_csv(csv_path)

  
    summary = summarize_data(df)

    df = clean_data(df)


    visualize_correlation(df)
    visualize_distribution(df)
    animated_trend(df)

    generate_profile(df)

    df.to_csv("cleaned_data.csv", index=False)
    print("\n Cleaned data saved as 'cleaned_data.csv'")

    print("\n Analysis complete! Generated files:")
    print(" - cleaned_data.csv")
    print(" - correlation_heatmap.png")
    print(" - feature_distributions.png")
    print(" - animated_trend.gif")
    print(" - analysis_report.html")
    print("\n All outputs are in the current folder. ")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pro_analyze_csv.py <path_to_csv>")
    else:
        analyze_csv(sys.argv[1])

