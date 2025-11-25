import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, chi2_contingency, pearsonr
from scipy.stats import chi2_contingency, f_oneway, kruskal

# --- Configuration ---
# Set plotting style for consistency
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ----------------------------------------------------
# 1. UNIVARIATE ANALYSIS FUNCTIONS
# ----------------------------------------------------

def univariate_analysis_numerical(df, column):
    """Performs statistical and visual EDA for a numerical column."""
    print(f"--- Univariate Analysis: {column} (Numerical) ---")

    # A. Statistical Summary
    print("\nStatistical Summary:")
    print(df[column].describe())
    
    # B. Normality Test (for small datasets, optional)
    stat, p = shapiro(df[column].dropna())
    print(f"\nShapiro-Wilk Test: Stat={stat:.3f}, p-value={p:.3f}")
    if p < 0.05:
        print("-> The data is likely not normally distributed (reject H0).")
    
    # C. Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True, color='skyblue')
    plt.title(f'Distribution of {column} (KDE/Histogram)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[column], color='salmon')
    plt.title(f'Box Plot of {column} (Outliers/Spread)')
    
    plt.tight_layout()
    plt.show()

def univariate_analysis_categorical(df, column, top_n_plot=20):
    """
    Performs fast statistical and visual EDA for a categorical column, 
    optimized for large datasets by handling high cardinality and frequency 
    using standard print statements.
    """
    data_size = len(df)
    
    print(f"\n--- Univariate Analysis: {column} (Categorical) ---")

    # A. Statistical Summary (Counts and Proportions)
    # Use dropna=False to explicitly show missing values count
    counts = df[column].value_counts(dropna=False)
    proportions = df[column].value_counts(normalize=True, dropna=False) * 100
    
    print("\nStatistical Summary (Top 10 Value Counts):")
    print(counts.head(10))
    print("\nProportions (%):\n")
    print(proportions.head(10).round(2))

    # Identify number of unique values
    unique_count = len(counts)
    
    # B. Visualization Optimization
    
    # High Cardinality Check: Only plot top N if unique count is too large
    if unique_count > top_n_plot + 5: # Allow a small buffer
        
        categories_to_plot = counts.head(top_n_plot).index.tolist()
        
        # Create a temporary column for plotting: group infrequent categories into 'Other'
        # This prevents performance issues and visual clutter from plotting hundreds of bars.
        df_plot_series = df[column].apply(
            lambda x: x if x in categories_to_plot else 'Other'
        )
        
        plot_data = df_plot_series.value_counts()
        plot_title = f'Frequency of {column} (Top {len(categories_to_plot)} Categories + Other)'
        
        print(f"\n[WARNING] Column has high cardinality ({unique_count} unique values). Plotting only Top {len(categories_to_plot)} categories + 'Other'.")

    else:
        # Plot all categories if manageable
        plot_data = counts
        plot_title = f'Frequency of {column}'
        
    # Final Plotting
    # Dynamic figure size: Height adjusts based on the number of bars (max 10 inches)
    plt.figure(figsize=(10, min(10, len(plot_data) * 0.4 + 2)))
    
    # Use sns.barplot for efficiency, feeding it pre-calculated counts
    sns.barplot(x=plot_data.values, y=plot_data.index, palette='viridis', orient='h')
    
    # Add percentage labels 
    total = plot_data.sum()
    for i, v in enumerate(plot_data.values):
        plt.text(v + (total * 0.005), i, f'{v/total*100:.1f}%', va='center')
        
    plt.title(plot_title)
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()
    print("-" * 50)

# ----------------------------------------------------
# 2. BIVARIATE ANALYSIS FUNCTIONS
# ----------------------------------------------------
def bivariate_num_vs_num(df, col1, col2, plot_type='hex'):
    """
    Optimized Bivariate analysis for two numerical columns.
    Handles NaN mismatch automatically and uses Hexbin/KDE plots for large data.
    """
    print(f"\n--- Bivariate Analysis: {col1} vs {col2} ---")
    
    # 1. Statistical Test (Efficient and Safe Pearson Correlation)
    
    # Use Pandas .corr() on the subset. It correctly handles pairwise dropping of NaNs.
    corr_matrix = df[[col1, col2]].corr(method='pearson')
    corr = corr_matrix.iloc[0, 1]
    
    # Calculate p-value only on complete cases for accuracy
    temp_df = df[[col1, col2]].dropna()
    
    if len(temp_df) < 2:
        print("[WARNING] Not enough non-missing data points for correlation calculation.")
        return
        
    # Get the p-value using scipy on the cleaned subset
    _, p_value = pearsonr(temp_df[col1], temp_df[col2])
    
    print(f"\nPearson Correlation (r): {corr:.3f}, p-value: {p_value:.3e} (N={len(temp_df)})")
    
    # 2. Meaningful Insight
    if p_value < 0.05:
        strength = "strong" if abs(corr) >= 0.7 else ("moderate" if abs(corr) >= 0.3 else "weak")
        direction = "positive" if corr > 0 else "negative"
        print(f"-> INSIGHT: There is a **significant, {strength} {direction} linear relationship** between {col1} and {col2}.")
    else:
        print(f"-> INSIGHT: **No statistically significant linear relationship** found between {col1} and {col2}.")

    # 3. Visualization Optimization (Hexbin or KDE for large data)
    plt.figure(figsize=(10, 8))
    
    if plot_type == 'hex':
        # Hexbin plots efficiently visualize density by binning points, avoiding slow scatter rendering
        plt.hexbin(df[col1].dropna(), df[col2].dropna(), gridsize=50, cmap='viridis')
        plt.colorbar(label='Count in Bin')
        plt.title(f'Hexbin Plot (Density) of {col1} vs {col2}')
        print(f"\n[INFO] Using Hexbin plot for efficient visualization of dense data.")

    elif plot_type == 'kde':
        # KDE plots show smoothed density contours, better for non-linear patterns
        sns.kdeplot(x=col1, y=col2, data=df, fill=True, cmap='plasma')
        plt.title(f'KDE Plot (Density Contour) of {col1} vs {col2}')
        print(f"\n[INFO] Using KDE plot for density visualization.")
        
    else: # Default back to scatter for smaller N or if requested
        sns.scatterplot(x=col1, y=col2, data=df, alpha=0.3) # Lower alpha for large N scatter
        plt.title(f'Scatter Plot of {col1} vs {col2} (Alpha=0.3)')
    
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()
    print("-" * 50)


# --- 

def bivariate_num_vs_cat(df, num_col, cat_col):
    """
    Optimized Bivariate analysis for Numerical vs. Categorical column. 
    Uses ANOVA for statistical insight and Boxenplot for efficient visualization on large data.
    """
    print(f"\n--- Bivariate Analysis: {num_col} by {cat_col} ---")
    
    # 1. Statistical Insight (ANOVA/Kruskal-Wallis Test)
    # Check for group sizes and number of groups (efficiency)
    groups = [df[num_col].loc[df[cat_col] == cat].dropna() for cat in df[cat_col].unique() if len(df[num_col].loc[df[cat_col] == cat].dropna()) > 0]
    
    if len(groups) > 1:
        try:
            # ANOVA (requires approximate normal distribution and equal variance)
            f_stat, p_value = f_oneway(*groups)
            test_name = "One-way ANOVA"
        except ValueError:
            # Fallback to Kruskal-Wallis (non-parametric, safer for real-world data)
            f_stat, p_value = kruskal(*groups)
            test_name = "Kruskal-Wallis H-test"
            
        print(f"\nStatistical Test: {test_name}")
        print(f"F-statistic/H-statistic: {f_stat:.3f}, p-value: {p_value:.3e}")
        
        # Meaningful Insight
        if p_value < 0.05:
            print(f"-> INSIGHT: The mean/median of **{num_col}** is **significantly different** across the categories of **{cat_col}** (Reject H0).")
        else:
            print(f"-> INSIGHT: There is **no significant difference** in {num_col} means/medians across categories.")
    else:
        print("\n[WARNING] Not enough non-empty groups to perform a statistical test.")

    # 2. Optimized Statistical Summary (Mean, Median, Std)
    summary = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std']).sort_values(by='mean', ascending=False)
    print(f"\nGrouped Mean, Median, and Std Dev of {num_col}:")
    print(summary)

    # 3. Visualization Optimization (Boxenplot)
    # Boxenplot (or letter-value plot) is faster and more effective than ViolinPlot for large N
    plt.figure(figsize=(12, 6))
    
    # Only plot the top 10 categories to avoid clutter and improve speed if cardinality is high
    top_categories = df[cat_col].value_counts().head(10).index
    df_plot = df[df[cat_col].isin(top_categories)]

    sns.boxenplot(x=cat_col, y=num_col, data=df_plot)
    plt.title(f'{num_col} Distribution by {cat_col} Category (Top {len(top_categories)})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ---
# ---

def bivariate_cat_vs_cat(df, col1, col2, top_n_plot=10):
    """
    Optimized Bivariate analysis for two Categorical columns.
    Uses Chi-Squared Test and Normalized Bar Plot for insightful comparison.
    """
    print(f"\n--- Bivariate Analysis: {col1} vs {col2} ---")
    
    # 1. Statistical Test (Chi-Squared Test for independence)
    crosstab = pd.crosstab(df[col1], df[col2])
    
    # Handle high cardinality for the test by only using the most common categories
    if len(crosstab.index) > top_n_plot or len(crosstab.columns) > top_n_plot:
        crosstab_test = pd.crosstab(df[col1], df[col2]).head(top_n_plot).iloc[:, :top_n_plot]
        print(f"[WARNING] Test truncated to Top {top_n_plot} rows/cols for efficiency.")
    else:
        crosstab_test = crosstab

    chi2, p, dof, expected = chi2_contingency(crosstab_test)
    print(f"\nChi-Squared Test: Chi2={chi2:.3f}, p-value={p:.3e}, DOF={dof}")
    
    # Meaningful Insight
    if p < 0.05:
        print(f"-> INSIGHT: There is a **statistically significant association** between **{col1}** and **{col2}** (Reject H0).")
    else:
        print(f"-> INSIGHT: **{col1}** and **{col2}** appear to be statistically independent.")

    # 2. Optimized Visualization (Normalized Bar Plot)
    # Stacked count plots are hard to interpret on large N. Use normalization.
    plt.figure(figsize=(12, 7))
    
    # Normalize the crosstab by the rows (col1) to show proportions
    normalized_crosstab = crosstab.div(crosstab.sum(axis=1), axis=0)
    
    # Only plot the top 10 rows for readability
    normalized_crosstab.head(top_n_plot).plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis', ax=plt.gca())

    plt.title(f'Normalized Distribution of {col2} within {col1} (Top {top_n_plot} Categories)')
    plt.ylabel(f'Proportion of {col2} (Sum to 1.0 per {col1} category)')
    plt.xlabel(col1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=col2)
    plt.tight_layout()
    plt.show()

    # 3. Final Summary Table
    print("\nCrosstab (Raw Counts, Top 10x10):")
    print(crosstab.head(10).iloc[:, :10])

# ----------------------------------------------------
# 3. MULTIVARIATE ANALYSIS FUNCTION
# ----------------------------------------------------

def multivariate_analysis(df, numerical_cols, hue_col=None, pairplot_sampling_n=2000):
    """
    Performs optimized multivariate analysis for large datasets.
    Uses efficient correlation and subsampling for the Pair Plot.
    """
    print("\n--- Multivariate Analysis: Correlation and Interactions ---")
    
    # Check for memory safety before expensive operations
    if len(df) > 500000:
        print("[WARNING] Dataset is very large. Pair Plot may be slow or omitted.")

    # A. Correlation Heatmap (for Numerical Columns) - Highly Efficient
    # This remains fast as it only depends on the number of columns (N), not rows (M).
    print("\n1. Correlation Heatmap:")
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5,
        cbar_kws={'shrink': 0.8} # Improve color bar readability
    )
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
    plt.show()

    # B. Meaningful Insight from Correlation
    high_corr_pairs = []
    # Find absolute correlation > 0.7 (or any chosen threshold)
    for i in range(len(numerical_cols)):
        for j in range(i + 1, len(numerical_cols)):
            col1 = numerical_cols[i]
            col2 = numerical_cols[j]
            corr_val = corr_matrix.loc[col1, col2]
            if abs(corr_val) >= 0.7:
                high_corr_pairs.append((col1, col2, corr_val))
    
    if high_corr_pairs:
        print("\n-> INSIGHT: High Multicollinearity Detected (Corr >= 0.7):")
        for c1, c2, r in high_corr_pairs:
            print(f"   - {c1} vs {c2}: r={r:.2f}. Consider dropping one for modeling.")
    else:
        print("\n-> INSIGHT: No highly correlated pairs (r < 0.7) found among numerical features.")

    print("\n" + "="*50)

    # C. Optimized Pair Plot (Focus on Subsampling)
    # Pair Plot is O(M * N^2) - must be limited for large M (rows)
    
    if len(numerical_cols) > 7:
        # If too many columns, Pair Plot is too large to display use fewer columns
        print(f"[WARNING] Skipping full Pair Plot. Too many features ({len(numerical_cols)}).")
        return

    if len(df) > pairplot_sampling_n:
        # 1. Subsample the data for plotting efficiency
        df_plot = df[numerical_cols + ([hue_col] if hue_col else [])].sample(
            n=pairplot_sampling_n, 
            random_state=42
        )
        print(f"2. Pair Plot (Subsampled): Plotting {pairplot_sampling_n} random rows for speed.")
    else:
        df_plot = df[numerical_cols + ([hue_col] if hue_col else [])].copy()
        print("2. Pair Plot: Plotting all data.")

    # 2. Plotting (Using Hexbin/KDE on off-diagonals for density insight)
    g = sns.pairplot(
        df_plot, 
        hue=hue_col, 
        corner=True,
        diag_kind='kde', # Use KDE for diagonal density
        plot_kws={'alpha': 0.6, 's': 20} # Smaller points, lower alpha for better visibility
    )
    # The off-diagonal plots in PairPlot use scatter. 
    # For large N, consider manually replacing them with hexbin/kde via g.map_offdiag
    
    g.fig.suptitle("Pair Plot Matrix (Subsampled)", y=1.02, fontsize=16)
    plt.show()
    print("-" * 50)

# ----------------------------------------------------
# 4. TEMPLATE EXECUTION (How to Use)
# ----------------------------------------------------

def run_eda_template(df, numerical_features, categorical_features, target_col):
    """Master function to run the full suite of EDA."""
    print("================== Starting Full EDA Pipeline ==================")

    # 1. Run Univariate Analysis
    for col in numerical_features:
        univariate_analysis_numerical(df, col)
    for col in categorical_features:
        univariate_analysis_categorical(df, col)
    
    # 2. Run Bivariate Analysis (Key interactions, typically against the Target)
    print("\n================== Key Bivariate Analysis (vs. Target) ==================")
    
    # Numerical features vs. Categorical Target (if Target is categorical)
    for col in numerical_features:
        if target_col in categorical_features:
            bivariate_num_vs_cat(df, num_col=col, cat_col=target_col)
    
    # Categorical features vs. Categorical Target
    for col in categorical_features:
        if col != target_col:
            bivariate_cat_vs_cat(df, col1=col, col2=target_col)

    # 3. Run Multivariate Analysis
    multivariate_analysis(df, numerical_cols=numerical_features, hue_col=target_col)