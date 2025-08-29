import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

class DashboardWithInsights:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.insights = []

    # ---------- AI-Like Insight Generator ----------
    def generate_insights(self):
        insights = []

        # Example: Numerical column statistics
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            mean_val = self.data[col].mean()
            max_val = self.data[col].max()
            min_val = self.data[col].min()

            insights.append(f"📊 Column **{col}** → Mean: {mean_val:.2f}, Range: {min_val} to {max_val}")

            # Example AI-style reasoning
            if mean_val > (self.data[col].median()):
                insights.append(f"🔎 The average of {col} is above the median → possible positive skew.")
            else:
                insights.append(f"🔎 The average of {col} is below the median → possible negative skew.")

        # Example: Categorical column insights
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            top_cat = self.data[col].mode()[0]
            freq = self.data[col].value_counts().iloc[0]
            insights.append(f"📌 Most common category in {col} is '{top_cat}' with {freq} occurrences.")

        self.insights = insights
        return insights

    # ---------- Dashboard Generator ----------
    def generate_dashboard(self, output_file="dashboard_with_insights.png"):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Plot 1: Correlation Heatmap
        if not self.data.select_dtypes(include=['float64', 'int64']).empty:
            corr = self.data.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[0])
            axes[0].set_title("Correlation Heatmap")
        else:
            axes[0].text(0.5, 0.5, "No numeric data", ha="center", va="center")

        # Plot 2: Histogram of first numeric column
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            sns.histplot(self.data[numeric_cols[0]], bins=20, kde=True, ax=axes[1])
            axes[1].set_title(f"Distribution of {numeric_cols[0]}")
        else:
            axes[1].text(0.5, 0.5, "No numeric column", ha="center", va="center")

        # Plot 3: Countplot of first categorical column
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            sns.countplot(y=self.data[categorical_cols[0]], ax=axes[2])
            axes[2].set_title(f"Category Counts: {categorical_cols[0]}")
        else:
            axes[2].text(0.5, 0.5, "No categorical column", ha="center", va="center")

        # Plot 4: AI-Generated Insights
        self.generate_insights()  # Ensure insights are created
        insights_text = "\n".join(["\n".join(wrap(i, 60)) for i in self.insights[:8]])  # limit to 8
        axes[3].axis("off")
        axes[3].text(0, 1, "AI-Generated Insights", fontsize=14, weight="bold")
        axes[3].text(0, 0.9, insights_text, fontsize=11, va="top")

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"✅ Dashboard with insights saved as {output_file}")
