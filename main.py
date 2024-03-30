        
import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd
import numpy as np
from scipy.stats import skew

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="LIDA: Automatic Generation of Visualizations and Infographics",
    page_icon="📊",
)

st.write("# LIDA: Automatic Generation of Visualizations and Infographics using Large Language Models 📊")

st.sidebar.write("## Setup")

# Step 1 - Get OpenAI API key
openai_key = os.env.API_KEY 

if not openai_key:
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
    if openai_key:
        display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
        st.sidebar.write(f"Current key: {display_key}")
    else:
        st.sidebar.write("Please enter OpenAI API key.")
else:
    display_key = openai_key[:2] + "*" * (len(openai_key) - 5) + openai_key[-3:]
    st.sidebar.write(f"OpenAI API key loaded from environment variable: {display_key}")

st.markdown(
    """
    LIDA is a library for generating data visualizations and data-faithful infographics.
    LIDA is grammar agnostic (will work with any programming language and visualization
    libraries e.g. matplotlib, seaborn, altair, d3 etc) and works with multiple large language
    model providers (OpenAI, Azure OpenAI, PaLM, Cohere, Huggingface). Details on the components
    of LIDA are described in the [paper here](https://arxiv.org/abs/2303.02927) and in this
    tutorial [notebook](notebooks/tutorial.ipynb). See the project page [here](https://microsoft.github.io/lida/) for updates!.

   This demo shows how to use the LIDA python api with Streamlit. [More](/about).

   ----
""")


# Step 2 - Select a dataset and summarization method
if openai_key:
    # Initialize selected_dataset to None
    selected_dataset = None

    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    st.sidebar.write("## Text Generation Model")
    models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
    selected_model = st.sidebar.selectbox(
        'Choose a model',
        options=models,
        index=0
    )

    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0)

    # set use_cache in sidebar
    use_cache = st.sidebar.checkbox("Use cache", value=True)

    # Handle dataset selection and upload
    st.sidebar.write("## Data Summarization")
    st.sidebar.write("### Choose a dataset")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Cars", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
        {"label": "Weather", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    ]

    selected_dataset_label = st.sidebar.selectbox(
        'Choose a dataset',
        options=[dataset["label"] for dataset in datasets],
        index=0
    )
    data=None
    upload_own_data = st.sidebar.checkbox("Upload your own data")

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

        if uploaded_file is not None:
            # Get the original file name and extension
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Load the data depending on the file type
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            # Save the data using the original file name in the data dir
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})

            # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    else:
        selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]

    if not selected_dataset:
        st.info("To continue, select a dataset from the sidebar on the left or upload your own.")

    st.sidebar.write("### Choose a summarization method")
    summarization_methods = [
        {"label": "llm",
         "description":
         "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
        {"label": "default",
         "description": "Uses dataset column statistics and column names as the summary"},
        {"label": "columns", "description": "Uses the dataset column names as the summary"}]

    selected_method_label = st.sidebar.selectbox(
        'Choose a method',
        options=[method["label"] for method in summarization_methods],
        index=0
    )

    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)
        
        
# Step 3 - Generate data summary
if openai_key and selected_dataset and selected_method:
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)

         
    # Show all  data
    st.write("## Data")
    st.write(data)

    st.write("## Summary")
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))
        
        
    st.sidebar.write("## Data Cleaning")
    # Check if user wants to clean or fill data
    clean_data = st.sidebar.checkbox("Drop data")
    fill_data = st.sidebar.checkbox("Fill data")


    if clean_data:

        # Check for missing values and duplicates in uploaded data
        missing_values_count = data.isna().sum().sum()
        missing_values_df = data.isna().sum().to_frame('Missing Values').reset_index()
        duplicate_counts = data.duplicated().sum()
        column_duplicates = data.duplicated(subset=data.columns).sum(axis=0)
        column_duplicates_df = pd.DataFrame({'Column': ['Total'], 'Duplicate Count': [column_duplicates]})

        # Display information about missing values and duplicates in uploaded data
        st.write(f"The uploaded data contains {missing_values_count} missing values.")
        if isinstance(missing_values_df, pd.Series):
            missing_values_df = pd.Series(data.isna().sum()).to_frame('Missing Values').reset_index()

        st.dataframe(missing_values_df.rename(columns={'index': 'Column'}))
        st.write(f"The uploaded data contains {duplicate_counts} total duplicate rows.")
        st.write("Column-wise duplicate counts:")
        st.write(column_duplicates_df)

        # Clean data by dropping missing values
        cleaned_data = data.dropna()
        # Show all cleaned data
        st.write("## All Cleaned Data")
        st.write(cleaned_data)
        selected_dataset = cleaned_data

       

    elif fill_data:
   
        # Check for missing values and duplicates in uploaded data
        missing_values_count = data.isna().sum().sum()
        missing_values_df = data.isna().sum().to_frame('Missing Values').reset_index()
        duplicate_counts = data.duplicated().sum()
        column_duplicates = data.duplicated(subset=data.columns).sum(axis=0)
        column_duplicates_df = pd.DataFrame({'Column': ['Total'], 'Duplicate Count': [column_duplicates]})

        # Display information about missing values and duplicates in uploaded data
        st.write(f"The uploaded data contains {missing_values_count} missing values.")
        if isinstance(missing_values_df, pd.Series):
            missing_values_df = pd.Series(data.isna().sum()).to_frame('Missing Values').reset_index()

        st.dataframe(missing_values_df.rename(columns={'index': 'Column'}))
        st.write(f"The uploaded data contains {duplicate_counts} total duplicate rows.")
        st.write("Column-wise duplicate counts:")
        st.write(column_duplicates_df)

        # Impute missing values for numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                if skew(data[col]) > 1:  # Skewed data, use median
                    data[col].fillna(data[col].median(), inplace=True)
                else:  # Normal data, use mean
                    data[col].fillna(data[col].mean(), inplace=True)

        # Impute missing values for categorical columns
        categorical_cols = data.select_dtypes(exclude=np.number).columns
        for col in categorical_cols:
            if data[col].isnull().any():
                data[col].fillna(data[col].mode()[0], inplace=True)

        filled_data = data

        # Recalculate missing value and duplicate counts
        missing_values_count = filled_data.isna().sum().sum()
        missing_values_df = filled_data.isna().sum().to_frame('Missing Values').reset_index()
        duplicate_counts = filled_data.duplicated().sum()
        column_duplicates = filled_data.duplicated(subset=data.columns).sum(axis=0)
        column_duplicates_df = pd.DataFrame({'Column': ['Total'], 'Duplicate Count': [column_duplicates]})

        st.write(f"The filled data contains {missing_values_count} missing values.")
        st.dataframe(missing_values_df.rename(columns={'index': 'Column'}))
        st.write(f"The filled data contains {duplicate_counts} total duplicate rows.")
        st.write("Column-wise duplicate counts:")
        st.write(column_duplicates_df)

        selected_dataset = filled_data

        # Show all filled data
        st.write("## All Filled Data")
        st.write(filled_data)

    # Step 4 - Generate goals
    if summary:
        st.sidebar.write("### Goal Selection")

        num_goals = st.sidebar.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.sidebar.checkbox("Add Your Own Goal")

        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
        st.write(f"## Goals ({len(goals)})")

        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input("Describe Your Goal")

            if user_goal:

                new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)

        selected_goal_index = goal_questions.index(selected_goal)
        st.write(goals[selected_goal_index])

        selected_goal_object = goals[selected_goal_index]

    
      # Step 5 - Generate visualizations
    if selected_goal_object:
        st.sidebar.write("## Visualization Library")
        visualization_libraries = ["seaborn", "matplotlib", "plotly"]

        selected_library = st.sidebar.selectbox(
            'Choose a visualization library',
            options=visualization_libraries,
            index=0
        )

        st.write("## Visualizations")

        if clean_data:
            data_for_visualization = selected_dataset
            
        elif fill_data:
            data_for_visualization = selected_dataset
    
        else:
            data_for_visualization = data

        if data_for_visualization is not None and not data_for_visualization.empty:
            num_visualizations = st.sidebar.slider(
                "Number of visualizations to generate",
                min_value=1,
                max_value=10,
                value=2)

            textgen_config = TextGenerationConfig(
                n=num_visualizations, temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            if clean_data:
                visualizations = lida.visualize(
                    summary=summary,
                    goal=selected_goal_object,
                    textgen_config=textgen_config,
                    library=selected_library
                    )
            else:
                visualizations = lida.visualize(
                    summary=summary,
                    goal=selected_goal_object,
                    textgen_config=textgen_config,
                    library=selected_library)

            viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]

            if visualizations:
                st.write("Visualizations available:", len(visualizations))
                st.write("Viz Titles:", viz_titles)
                selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)
                selected_viz = visualizations[viz_titles.index(selected_viz_title)]

                if selected_viz.raster:
                    from PIL import Image
                    import io
                    import base64

                    imgdata = base64.b64decode(selected_viz.raster)
                    img = Image.open(io.BytesIO(imgdata))
                    st.image(img, caption=selected_viz_title, use_column_width=True)

                st.write("### Visualization Code")
                st.code(selected_viz.code)
            else:
                st.write("No visualizations available.")
        else:
            st.write("### Data is empty after cleaning.")
