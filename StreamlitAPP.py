import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
import streamlit as st

# load the response
with open(r"C:\Users\sampa\Desktop\New folder (3)\Gen_AI Ineuron Intelligence\mcqgen\response.json", "r") as file:
    RESPONSE_JSON = json.load(file)

# creating the title for the app
st.title("MCQ Creator Application with Langchain")

with st.form("user_inputs"):
    # file upload
    uploaded_file = st.file_uploader("upload a PSD or text file")

    # input fields
    mcq_count = st.number_input("No. of MCQs",min_value=3, max_value=50)

    # subject
    subject = st.text_input("insert subject", max_chars=20)

    # quiz tone
    tone = st.text_input("complexity of questions", max_chars=20, placeholder="Simple")

    # add button
    button = st.form_submit_button("Create MCQs")

    # check if button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading......."):
            try:
                text = read_file(uploaded_file)
                # count tokens and cost of the API
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")


            else:
                print(f"total tokens: {cb.total_tokens}")
                print(f"prompt tokens: {cb.prompt_tokens}")
                print(f"complete tokens: {cb.completion_tokens}")
                print(f"total_cost: {cb.total_cost}")

                if isinstance(response, dict):
                    #Extract the quiz data from the response
                    quiz = response.get("quiz",None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)

                            # display the review in text box as well
                            st.text_area(label = "Review", value = response["review"])
                        else:
                            st.error("Error in the table data")

                else:
                    st.write(response)
