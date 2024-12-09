from collections import defaultdict
from textblob import TextBlob
import nltk
# Download the vader_lexicon
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, TFDistilBertForSequenceClassification, DistilBertTokenizer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
import tensorflow as tf

# Streamlit app setup
st.title("NUSYLL Data Processing and Visualization Dashboard")

# Sidebar buttons
st.sidebar.header("Options")
option = st.sidebar.radio("Choose an action:", ("Upload Data", "Generate Excel Files", "Plot Charts"))

merged_file2 = pd.DataFrame()
min_rank = -1
max_rank = -1
min_pci = 0
max_pci = 0
min_domainID = -1
max_domainID = -1
min_globalScore = -1
max_globalScore = -1

# Global variables for storing data
if "file1" not in st.session_state:
    st.session_state["file1"] = None
if "file2" not in st.session_state:
    st.session_state["file2"] = None
if "merged_file" not in st.session_state:
    st.session_state["merged_file"] = None

# Specify the model name and revision for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "main"
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# Function to get sentiment
def get_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)

    # Get model predictions
    outputs = model(**inputs)

    # Apply softmax to get probabilities (optional for classification)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=-1).numpy()[0]

    # Map the output class to sentiment labels
    sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    return sentiment


# Cache to store sentiment results
sentiment_cache = defaultdict(str)

nltk.download('vader_lexicon')

# Initialize necessary models
sia = SentimentIntensityAnalyzer()
sentiment_analyzer = pipeline('sentiment-analysis')


# Define function for ensemble-based sentiment analysis
def get_sentiment_ensemble(text):
    if not text:  # Handle empty strings
        return 'Neutral'

    # VADER Sentiment
    vader_score = sia.polarity_scores(text)
    vader_sentiment = 'Positive' if vader_score['compound'] > 0.05 else 'Negative' if vader_score[
                                                                                          'compound'] < -0.05 else 'Neutral'

    # TextBlob Sentiment
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_sentiment = 'Positive' if textblob_polarity > 0 else 'Negative' if textblob_polarity < 0 else 'Neutral'

    # Hugging Face Sentiment
    #result = sentiment_analyzer(text)
    #huggingface_sentiment = 'Positive' if result[0]['label'] == 'LABEL_1' else 'Negative'

    # Majority vote to decide final sentiment
    sentiments = [vader_sentiment, textblob_sentiment]
    return max(set(sentiments), key=sentiments.count)


# Define function for sentiment analysis using VADER
def get_sentiment_vader(text):
    if not text:  # Handle empty strings
        return 'Neutral'

    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']

    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
def merge_data(file1, file2):
    if 'Candidate ID' in file2.columns:
        merged_file = pd.merge(file1, file2, left_on='Index No', right_on='Candidate ID', how='outer')
    else:
        merged_file = pd.merge(file2, file1, left_on='Index No', right_on='Candidate ID', how='outer')

    with st.spinner("Processing sentiment analysis..."):
        merged_file['Comment Sentiment'] = merged_file['Station Flag Comment'].fillna('').apply(lambda x: get_sentiment_ensemble(x))
    # Create a new column for the estimated amount based on income ranges
    merged_file['Estimated Amount'] = merged_file['Income'].map(income_mapping)
    merged_file['Family Members'] = pd.to_numeric(merged_file['Family Members'], errors='coerce')
    merged_file['Global Score'] = pd.to_numeric(merged_file['Global Score / SP Score'], errors='coerce')
    merged_file['Domain ID'] = pd.to_numeric(merged_file['Domain ID'], errors='coerce')
    merged_file['Final FSA'] = pd.to_numeric(merged_file['Final FSA'], errors='coerce')
    merged_file['UAS'] = pd.to_numeric(merged_file['UAS'], errors='coerce')
    merged_file['PCI'] = merged_file.apply(
        lambda row: row['Estimated Amount'] / row['Family Members'] if row['Family Members'] > 0 else None,
        axis=1
    )
    return merged_file


# Define income ranges and their corresponding estimated amounts
income_mapping = {
    'Less than $1000': 1000.00,
    '$1,000 - $1,999': 1499.50,
    '$2,000 - $2,999': 2499.50,
    '$3,000 - $3,999': 3499.50,
    '$4,000 - $4,999': 4499.50,
    '$5,000 - $5,999': 5499.50,
    '$6,000 - $7,999': 6999.50,
    '$8,000 - $9,999': 8999.50,
    '$10,000 - $11,999': 10999.50,
    '$12,000 - $15,999': 13999.50,
    '$16,000 - $19,999': 17999.50,
    '$20,000 and above': 20000.00
}

def filter_by_CAT_UAS(df):
    condition = (
            ((df['CAT'] == 'A') & (df['UAS'] == 90)) |
            ((df['CAT'] == 'NUSHS') & (df['UAS'] == 5)) |
            ((df['CAT'] == 'Poly') & (df['UAS'] == 4)) |
            ((df['CAT'] == 'IB') & (df['UAS'] == 45))
    )
    return df[condition]

def convert_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Filtered Data')
    processed_data = output.getvalue()
    return processed_data

def to_excel(data_dict):
    """Convert data to an Excel file with multiple tabs."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, data in data_dict.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


# Option 1: Upload Data
if option == "Upload Data":
    st.header("Upload Data Files")

    file1 = st.file_uploader("Upload First File", type=["csv", "xlsx"])
    file2 = st.file_uploader("Upload Second File", type=["csv", "xlsx"])

    if file1:
        if file1.name.endswith(".csv"):
            st.session_state["file1"] = pd.read_csv(file1).astype(str)
        else:
            st.session_state["file1"] = pd.read_excel(file1).astype(str)
        st.write("First File Uploaded Successfully:")
        st.write(st.session_state["file1"])
    if file2:
        if file2.name.endswith(".csv"):
            st.session_state["file2"] = pd.read_csv(file2).astype(str)
        else:
            st.session_state["file2"] = pd.read_excel(file2).astype(str)
        st.write("Second File Uploaded Successfully:")
        st.write(st.session_state["file2"])

        # Combine button and key selection
        if st.session_state["file1"] is not None and st.session_state["file2"] is not None:
            if st.button("Combine Data"):
                try:
                    merged_file2 = merge_data(
                        st.session_state["file1"],
                        st.session_state["file2"]
                    )
                    st.session_state["merged_file"] = merged_file2

                    st.success("Files Combined Successfully!")
                    st.write(st.session_state["merged_file"])
                except Exception as e:
                    st.error(f"Error combining files: {e}")

# Option 2: Generate Excel Files
elif option == "Generate Excel Files":
    st.header("Generate Excel Files")

    if st.session_state["file1"] is None and st.session_state["file2"] is None:
        st.warning("Please upload both data files first!")
    else:
        st.write("Filter and generate Excel files based on criteria:")
        # Filter Options
        if not st.session_state["merged_file"].empty:
            st.session_state["merged_file"]["Rank"] = pd.to_numeric(st.session_state["merged_file"]["Rank"], errors='coerce')
            st.session_state["merged_file"]["Domain ID"] = pd.to_numeric(st.session_state["merged_file"]["Domain ID"],errors='coerce')
            st.session_state["merged_file"]["Global Score"] = pd.to_numeric(st.session_state["merged_file"]["Global Score"], errors='coerce')
            st.session_state["merged_file"]["PCI"] = pd.to_numeric(st.session_state["merged_file"]["PCI"],errors='coerce')
            # Calculate min and max rank values dynamically
            min_rank = int(st.session_state["merged_file"]["Rank"].min())
            max_rank = int(st.session_state["merged_file"]["Rank"].max())
            min_domainID = int(st.session_state["merged_file"]["Domain ID"].min())
            max_domainID = int(st.session_state["merged_file"]["Domain ID"].max())
            min_globalScore = int(st.session_state["merged_file"]["Global Score"].min())
            max_globalScore = int(st.session_state["merged_file"]["Global Score"].max())
            min_pci = int(st.session_state["merged_file"]['PCI'].min())
            max_pci = int(st.session_state["merged_file"]['PCI'].max())

        with st.form("filter_form"):
            st.subheader("Rank Range")
            apply_rank_filter = st.checkbox("Apply Rank Range Filter", value=False)

            rank_options = [None] + list(range(min_rank, max_rank + 1))
            start_rank = st.selectbox("Select Start Rank", options=rank_options, index=0)
            end_rank = st.selectbox("Select End Rank", options=rank_options, index=0)

            st.subheader("PCI Range")
            apply_pci_filter = st.checkbox("Apply PCI Range Filter", value=False)
            min_pci_value = st.number_input("Enter Minimum PCI", value=min_pci, min_value=min_pci, max_value=max_pci,
                                            step=1)
            max_pci_value = st.number_input("Enter Maximum PCI", value=max_pci, min_value=min_pci, max_value=max_pci,
                                            step=1)

            if min_pci_value > max_pci_value:
                st.warning("Minimum PCI cannot be greater than Maximum PCI.")

            # CAT and UAS filters
            st.subheader("Filter CAT | UAS")
            apply_cat_filter = st.checkbox("Apply CAT Filters", value=False)  # Checkbox to enable/disable the filter

            default_uas_values = {'A': 90, 'NUSHS': 5, 'Poly': 4, 'IB': 45}
            cat_filter = st.multiselect("Select Categories (CAT):", options=default_uas_values.keys(), default=list(default_uas_values.keys()))
            uas_values = {}
            for cat in cat_filter:
                uas_values[cat] = st.number_input(
                    f"Enter the desired UAS value for {cat} (default: {default_uas_values[cat]}):",
                    min_value=0, value=default_uas_values[cat], step=1)
            st.subheader("Filter by Comment Sentiment")
            sentiment_filter = st.selectbox("Choose Comment Sentiment Category", ["All", "positive", "negative", "neutral"])

            domain_filter = st.multiselect("Select Domain ID", options=list(range(min_domainID, max_domainID + 1)))

            global_score_filter = st.selectbox("Select Global Score", ["All"]+list(range(min_globalScore, max_globalScore + 1)))
            st.subheader("Find Duplicate Ranks")
            find_duplicates = st.checkbox("Find Duplicate Ranks")

            submitted = st.form_submit_button("Apply Filters")

        # Other Filtering Options section (outside the form)
        st.subheader("Other Filtering Options")
        col_options = [None] + list(st.session_state["merged_file"].columns)
        col_name = st.selectbox("Select a column for filtering:", options=col_options, index=0)
        if col_name != None:
            unique_values = st.session_state["merged_file"][col_name].dropna().unique()
            selected_value = st.selectbox(
                "Select value for filtering",
                options=[None] if col_name is None else list(unique_values), index=0)
        additional_filtering = st.button("Filter")

        # Filter data
        filtered_data = st.session_state["merged_file"]

        if submitted:
            filtered_data = st.session_state["merged_file"]
            if start_rank and end_rank and start_rank > end_rank:
                st.warning("Start Rank cannot be greater than End Rank.")
            if apply_rank_filter and start_rank is not None and end_rank is not None:
                filtered_data = filtered_data[(filtered_data['Rank'] >= start_rank) & (filtered_data['Rank'] <= end_rank)]

            if apply_pci_filter and min_pci_value is not None and max_pci_value is not None:
                filtered_data = filtered_data[(filtered_data['PCI'] >= min_pci_value) & (filtered_data['PCI'] <= max_pci_value)]

            if find_duplicates:
                filtered_data = filtered_data[filtered_data.duplicated('Rank', keep=False)]
                if not filtered_data.empty:
                    st.subheader("Duplicate Ranks Found")
                else:
                    st.success("No duplicate ranks found!")

            if sentiment_filter != "All":
                filtered_data = filtered_data[filtered_data['Comment Sentiment'] == sentiment_filter]

            if apply_cat_filter:
                cat_filtered_data = filtered_data.dropna(subset=['CAT', 'UAS'])

                # Match CAT and UAS values with the user-provided or default values
                uas_df = pd.DataFrame(list(uas_values.items()), columns=['CAT', 'UAS'])
                filtered_data = cat_filtered_data.merge(uas_df, on=['CAT', 'UAS'], how='inner')

            if domain_filter:
                filtered_data = filtered_data[filtered_data['Domain ID'].isin(domain_filter)]

            if global_score_filter != "All":
                filtered_data = filtered_data[filtered_data["Global Score"].isin(global_score_filter)]

        if additional_filtering:
            if col_name != None:
                filtered_data = filtered_data[filtered_data[col_name] == selected_value]
            else:
                st.warning("No column selected.")

        # Display filtered data
        if not filtered_data.empty:
            st.subheader("Filtered Data Preview")
            st.dataframe(filtered_data)

            # Create Excel
            if st.button("Generate Excel File"):
                filename = st.text_input("Enter filename (without extension):", value="filtered_data")
                if filename:
                    excel_data = convert_to_excel(filtered_data)
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name=f"{filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("Please provide a valid filename!")
        else:
            st.warning("No data matches the selected filters!")

# Option 3: Plot Charts
elif option == "Plot Charts":
    st.header("Plot Charts")

    if st.session_state["file1"] is None or st.session_state["file2"] is None:
        st.warning("Please upload both data files first!")
    else:
        st.write("Choose columns to plot:")
        x_col = st.selectbox("Select X-axis column", options=st.session_state["merged_file"].columns)

        y_aggregation_options = ["None", "Default: Count"]
        numeric_columns = [
            col for col in st.session_state["merged_file"].columns
            if pd.api.types.is_numeric_dtype(st.session_state["merged_file"][col])
        ]
        y_cols = st.multiselect(
            "Select up to 4 Y-axis columns/Aggregation to plot:",
            options=y_aggregation_options + numeric_columns, max_selections=4)
        if y_cols:
            reset_button = st.button("View Default Charts")
            if reset_button:
                y_cols = []
                st.session_state["merged_file"] = st.session_state["merged_file"]

        if (not y_cols or x_col is None):
            tabs = st.tabs(["Bar Charts", "Distribution Bar Charts"])
            with tabs[0]:
                # First Tab: Default Bar Charts
                st.subheader("Default Bar Charts")
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))

                # Default Bar Plot for Qualification vs FSA (mean scores)
                qualification_avg = st.session_state["merged_file"].groupby('CAT')['Final FSA'].mean()
                sns.barplot(x=qualification_avg.index, y=qualification_avg.values, palette='Blues', ax=axs[0, 0])
                axs[0, 0].set_title('Average FSA Scores by Qualification', fontsize=16, fontweight='bold')
                axs[0, 0].set_xlabel('Qualification', fontsize=14)
                axs[0, 0].set_ylabel('Mean FSA Score', fontsize=14)
                axs[0, 0].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[0, 0])

                # Default Bar Plot for Top 10 Schools by Mean FSA Score
                top_10_schools = st.session_state["merged_file"].groupby('School Name')['Final FSA'].mean().nlargest(10)
                sns.barplot(x=top_10_schools.index, y=top_10_schools.values, palette='viridis', ax=axs[0, 1])
                axs[0, 1].set_title('Top 10 Schools by Mean FSA Score', fontsize=16, fontweight='bold')
                axs[0, 1].set_xlabel('School Name', fontsize=14)
                axs[0, 1].set_ylabel('Mean FSA Score', fontsize=14)
                axs[0, 1].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[0, 1])

                # Default Bar Plot for Race vs FSA (mean scores)
                race_avg = st.session_state["merged_file"].groupby('Race_t')['Final FSA'].mean()
                sns.barplot(x=race_avg.index, y=race_avg.values, palette='Set2', ax=axs[1, 0])
                axs[1, 0].set_title('FSA Scores by Race', fontsize=16, fontweight='bold')
                axs[1, 0].set_xlabel('Race', fontsize=14)
                axs[1, 0].set_ylabel('Mean FSA Score', fontsize=14)
                axs[1, 0].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[1, 0])

                # Default Bar Plot for Gender vs FSA (mean scores)
                gender_avg = st.session_state["merged_file"].groupby('Gender')['Final FSA'].mean()
                sns.barplot(x=gender_avg.index, y=gender_avg.values, palette='coolwarm', ax=axs[1, 1])
                axs[1, 1].set_title('FSA Scores by Gender', fontsize=16, fontweight='bold')
                axs[1, 1].set_xlabel('Gender', fontsize=14)
                axs[1, 1].set_ylabel('Mean FSA Score', fontsize=14)
                axs[1, 1].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[1, 1])

                plt.tight_layout()
                st.pyplot(fig)
            with tabs[1]:
                # Second Tab: Distribution Bar Charts
                st.subheader("Distribution Bar Charts")
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))

                qualification_counts = st.session_state["merged_file"]['CAT'].value_counts()
                sns.barplot(x=qualification_counts.index, y=qualification_counts.values, palette='Blues', ax=axs[0, 0])
                axs[0, 0].set_title('Qualification Distribution', fontsize=18, fontweight='bold')
                axs[0, 0].set_xlabel('Qualification', fontsize=14)
                axs[0, 0].set_ylabel('Count', fontsize=14)
                axs[0, 0].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[0, 0])

                # Distribution for School
                school_counts = st.session_state["merged_file"]['School Name'].value_counts().nlargest(10)
                sns.barplot(x=school_counts.index, y=school_counts.values, palette='viridis', ax=axs[0, 1])
                axs[0, 1].set_title('School Distribution', fontsize=18, fontweight='bold')
                axs[0, 1].set_xlabel('School', fontsize=14)
                axs[0, 1].set_ylabel('Count', fontsize=14)
                axs[0, 1].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[0, 1])

                # Distribution for Race
                race_counts = st.session_state["merged_file"]['Race_t'].value_counts()
                sns.barplot(x=race_counts.index, y=race_counts.values, palette='Set2', ax=axs[1, 0])
                axs[1, 0].set_title('Race Distribution', fontsize=18, fontweight='bold')
                axs[1, 0].set_xlabel('Race', fontsize=14)
                axs[1, 0].set_ylabel('Count', fontsize=14)
                axs[1, 0].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[1, 0])

                # Distribution for Gender

                gender_counts = st.session_state["merged_file"]['Gender'].value_counts()
                sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='coolwarm', ax=axs[1, 1])
                axs[1, 1].set_title('Gender Distribution', fontsize=18, fontweight='bold')
                axs[1, 1].set_xlabel('Gender', fontsize=14)
                axs[1, 1].set_ylabel('Count', fontsize=14)
                axs[1, 1].tick_params(axis='x', rotation=45)
                sns.despine(ax=axs[1, 1])

                plt.tight_layout()
                st.pyplot(fig)

        else:
            if len(y_cols) == 0:
                st.warning("Please select at least one Y-axis column to proceed.")
            else:
                tabs = st.tabs([f"Chart {i + 1}" for i in range(len(y_cols))])

                for i, y_col in enumerate(y_cols):
                    with tabs[i]:
                        st.subheader(f"Chart for {y_col} by {x_col}")

                        # Aggregation method for numeric Y-axis
                        aggregation = None
                        if y_col in numeric_columns:
                            aggregation = st.selectbox(
                                f"Select Aggregation Method for {y_col}",
                                options=["Sum", "Average", "Median", "Minimum", "Maximum"],
                                key=f"agg_{y_col}"
                            )
                        else:
                            aggregation = y_col  # Default for "Count" option

                        # Chart type selection
                        chart_type = st.radio(
                            f"Select Chart Type for {y_col}",
                            options=["Bar Chart", "Line Chart"],
                            key=f"chart_type_{y_col}"
                        )

                        if st.button(f"Generate Chart for {y_col}", key=f"generate_{y_col}"):
                            # Prepare data
                            agg_methods = {
                                "Sum": "sum",
                                "Average": "mean",
                                "Median": "median",
                                "Minimum": "min",
                                "Maximum": "max"
                            }
                            if y_col in numeric_columns:
                                plot_data = st.session_state["merged_file"].groupby(x_col)[y_col].agg(agg_methods[aggregation])
                            else:
                                plot_data = st.session_state["merged_file"][x_col].value_counts()
                            # Generate and display chart
                            plt.figure(figsize=(8, 5))
                            if chart_type == "Bar Chart":
                                plot_data.plot(kind='bar', color='skyblue', edgecolor='black')
                                plt.title(f"Bar Chart: {y_col} ({aggregation}) by {x_col}")
                                plt.xlabel(x_col)
                                plt.ylabel(f"{y_col} ({aggregation})")
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                            elif chart_type == "Line Chart":
                                plot_data.plot(kind='line', marker='o', color='orange')
                                plt.title(f"Line Chart: {y_col} ({aggregation}) by {x_col}")
                                plt.xlabel(x_col)
                                plt.ylabel(f"{y_col} ({aggregation})")

                            st.pyplot(plt)


