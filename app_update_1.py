import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import json
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import base64

# Set page configuration
st.set_page_config(page_title="Advanced EDA App", layout="wide")


# Function to load data
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            # First, try to read the CSV normally
            try:
                df = pd.read_csv(file)
            except pd.errors.ParserError:
                # If that fails, try with error_bad_lines=False (for pandas < 1.3.0)
                try:
                    df = pd.read_csv(file, error_bad_lines=False, warn_bad_lines=True)
                except TypeError:
                    # For pandas >= 1.3.0, use on_bad_lines='warn'
                    df = pd.read_csv(file, on_bad_lines='warn')
                st.warning(
                    "Some lines in the CSV file could not be parsed correctly. The problematic rows have been skipped.")
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        elif file.name.endswith('.parquet'):
            df = pq.read_table(file).to_pandas()
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, JSON, or Parquet file.")
            return None

        # Validate for empty or corrupt files
        if df.empty:
            st.error("The uploaded file is empty. Please check your data.")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}. Please check if the file is corrupt or in the correct format.")
        st.info("If you're uploading a CSV file, try the following:")
        st.info("1. Check if the file uses a non-standard delimiter (e.g., semicolon instead of comma).")
        st.info("2. Ensure all rows have the same number of fields.")
        st.info("3. Look for and remove any extraneous quotation marks or commas within fields.")
        st.info("4. If the issue persists, try opening the file in a text editor to identify problematic rows.")
        return None


# Function to get summary statistics
def get_summary_stats(df, selected_columns):
    numeric_cols = df[selected_columns].select_dtypes(include=[np.number]).columns
    categorical_cols = df[selected_columns].select_dtypes(exclude=[np.number]).columns

    summary = pd.DataFrame({
        'Column': selected_columns,
        'Type': df[selected_columns].dtypes,
        'Non-Null Count': df[selected_columns].notnull().sum(),
        'Null Count': df[selected_columns].isnull().sum(),
        'Unique Values': df[selected_columns].nunique()
    })

    if len(numeric_cols) > 0:
        numeric_summary = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).T
        numeric_summary = numeric_summary.apply(pd.to_numeric, errors='coerce')
        summary = summary.merge(numeric_summary, left_on='Column', right_index=True, how='left')

    if len(categorical_cols) > 0:
        cat_summary_list = []
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            top_value = value_counts.index[0] if len(value_counts) > 0 else None
            top_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
            cat_summary_list.append({
                'Column': col,
                'top': top_value,
                'freq': top_freq
            })
        cat_summary = pd.DataFrame(cat_summary_list)
        summary = summary.merge(cat_summary, on='Column', how='left')

    return summary


# Function to plot histogram
def plot_histogram(df, column):
    fig = px.histogram(df, x=column, title=f'Histogram of {column}')
    return fig


# Function to plot box plot
def plot_boxplot(df, column):
    fig = px.box(df, y=column, title=f'Box Plot of {column}')
    return fig


# Function to plot scatter plot
def plot_scatter(df, x_column, y_column):
    fig = px.scatter(df, x=x_column, y=y_column, title=f'Scatter Plot: {x_column} vs {y_column}')
    return fig


# Function to plot correlation heatmap
def plot_correlation_heatmap(df, selected_columns):
    numeric_df = df[selected_columns].select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Heatmap')
    return fig


# Function to plot bar chart for categorical data
def plot_bar_chart(df, column):
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = ['Category', 'Count']
    fig = px.bar(value_counts, x='Category', y='Count', title=f'Bar Chart of {column}')
    return fig


# Function to plot pairplot
def plot_pairplot(df, selected_columns):
    numeric_df = df[selected_columns].select_dtypes(include=[np.number])
    fig = px.scatter_matrix(numeric_df, title='Pairplot')
    return fig


# Function to suggest data cleaning steps
def suggest_data_cleaning(df):
    suggestions = []

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        suggestions.append("Consider handling missing values:")
        for col, count in missing_values[missing_values > 0].items():
            suggestions.append(f"  - Column '{col}' has {count} missing values. Consider imputation or removal.")

    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            suggestions.append(
                f"Column '{col}' has {len(outliers)} potential outliers. Consider investigating or treating them.")

    # Check for skewed distributions
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            suggestions.append(
                f"Column '{col}' is highly skewed (skewness: {skewness:.2f}). Consider applying a transformation.")

    return suggestions


# Function for basic feature engineering
def perform_feature_engineering(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

    # Scale numerical columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Generate interaction features for numeric columns
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1:]:
            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]

    return df


# Function to generate EDA report
def generate_eda_report(df, selected_columns, summary_stats, cleaning_suggestions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Exploratory Data Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Dataset Overview
    story.append(Paragraph("Dataset Overview", styles['Heading2']))
    story.append(Paragraph(f"Number of Rows: {df.shape[0]}", styles['BodyText']))
    story.append(Paragraph(f"Number of Columns: {df.shape[1]}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Summary Statistics
    story.append(Paragraph("Summary Statistics", styles['Heading2']))
    summary_data = [summary_stats.columns.tolist()] + summary_stats.values.tolist()
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # Data Cleaning Suggestions
    story.append(Paragraph("Data Cleaning Suggestions", styles['Heading2']))
    for suggestion in cleaning_suggestions:
        story.append(Paragraph(suggestion, styles['BodyText']))
    story.append(Spacer(1, 12))

    # Generate the PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


# Function for basic predictive modeling
def perform_basic_modeling(df, target_column):
    # Prepare the data
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column])
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)

    # Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_mse = mean_squared_error(y_test, dt_predictions)
    dt_r2 = r2_score(y_test, dt_predictions)

    return {
        'Linear Regression': {'MSE': lr_mse, 'R2': lr_r2},
        'Decision Tree': {'MSE': dt_mse, 'R2': dt_r2}
    }


# Main Streamlit app
def main():
    st.title("Advanced Exploratory Data Analysis (EDA) App")
    st.write("Upload your dataset and get instant insights!")

    # Add tooltips and instructions
    with st.expander("How to use this app"):
        st.write("""
        1. Upload a CSV, Excel, JSON, or Parquet file using the file uploader below.
        2. Once uploaded, you'll see a preview of your data and summary statistics.
        3. Use the sidebar to select columns for analysis and choose different visualizations.
        4. Explore data cleaning suggestions and feature engineering options.
        5. Generate an EDA report or perform basic predictive modeling.
        6. Download the processed data, summary statistics, or EDA report as needed.
        """)

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json", "parquet"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            st.success("Data loaded successfully!")
            st.write(f"Shape of the dataset: {df.shape}")

            # Display raw data
            st.subheader("Raw Data Preview")
            st.write(df.head())

            # Column selection
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select columns for analysis", all_columns, default=all_columns)

            if selected_columns:
                # Summary statistics
                st.subheader("Summary Statistics")
                summary_stats = get_summary_stats(df, selected_columns)
                st.write(summary_stats)

                # Download summary statistics
                summary_csv = summary_stats.to_csv(index=False)
                st.download_button(
                    label="Download Summary Statistics",
                    data=summary_csv,
                    file_name="summary_statistics.csv",
                    mime="text/csv"
                )

                # Data Visualizations
                st.subheader("Data Visualizations")

                # Select columns for visualization
                numeric_cols = df[selected_columns].select_dtypes(include=[np.number]).columns
                categorical_cols = df[selected_columns].select_dtypes(exclude=[np.number]).columns

                # Univariate Analysis
                st.write("### Univariate Analysis")
                selected_column = st.selectbox("Select a column for univariate analysis:", selected_columns)

                if selected_column in numeric_cols:
                    st.write("#### Histogram")
                    st.plotly_chart(plot_histogram(df, selected_column))

                    st.write("#### Box Plot")
                    st.plotly_chart(plot_boxplot(df, selected_column))
                elif selected_column in categorical_cols:
                    st.write("#### Bar Chart")
                    st.plotly_chart(plot_bar_chart(df, selected_column))

                # Bivariate Analysis
                st.write("### Bivariate Analysis")
                if len(numeric_cols) >= 2:
                    x_column = st.selectbox("Select X-axis column:", numeric_cols)
                    y_column = st.selectbox("Select Y-axis column:", [col for col in numeric_cols if col != x_column])

                    st.write("#### Scatter Plot")
                    st.plotly_chart(plot_scatter(df, x_column, y_column))
                else:
                    st.write("Insufficient numeric columns for bivariate analysis.")

                # Correlation Heatmap
                if len(numeric_cols) > 1:
                    st.write("### Correlation Heatmap")
                    st.plotly_chart(plot_correlation_heatmap(df, numeric_cols))
                else:
                    st.write("Insufficient numeric columns for correlation heatmap.")

                # Pairplot
                if len(numeric_cols) > 1:
                    st.write("### Pairplot")
                    st.plotly_chart(plot_pairplot(df, numeric_cols))
                else:
                    st.write("Insufficient numeric columns for pairplot.")

                # Data Cleaning Suggestions
                st.subheader("Data Cleaning Suggestions")
                cleaning_suggestions = suggest_data_cleaning(df)
                if cleaning_suggestions:
                    for suggestion in cleaning_suggestions:
                        st.write(suggestion)
                else:
                    st.write("No specific data cleaning suggestions at this time.")

                # Feature Engineering
                st.subheader("Feature Engineering")
                if st.button("Perform Basic Feature Engineering"):
                    with st.spinner("Performing feature engineering..."):
                        df_engineered = perform_feature_engineering(df)
                        st.success("Feature engineering completed!")
                        st.write("Preview of engineered features:")
                        st.write(df_engineered.head())

                        # Download engineered data
                        csv = df_engineered.to_csv(index=False)
                        st.download_button(
                            label="Download Engineered Data",
                            data=csv,
                            file_name="engineered_data.csv",
                            mime="text/csv"
                        )

                # EDA Report Export
                st.subheader("EDA Report Export")
                if st.button("Generate EDA Report"):
                    with st.spinner("Generating EDA report..."):
                        pdf_buffer = generate_eda_report(df, selected_columns, summary_stats, cleaning_suggestions)
                        st.success("EDA report generated successfully!")

                        # Provide download link for the PDF
                        b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="eda_report.pdf">Download EDA Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                # Modeling Preview
                st.subheader("Modeling Preview")
                target_column = st.selectbox("Select target column for modeling:", numeric_cols)
                if st.button("Perform Basic Modeling"):
                    with st.spinner("Performing basic modeling..."):
                        model_results = perform_basic_modeling(df, target_column)
                        st.success("Basic modeling completed!")
                        st.write("Model Performance:")
                        for model, metrics in model_results.items():
                            st.write(f"{model}:")
                            st.write(f"  Mean Squared Error: {metrics['MSE']:.4f}")
                            st.write(f"  R-squared: {metrics['R2']:.4f}")

            else:
                st.warning("Please select at least one column for analysis.")


if __name__ == "__main__":
    main()