import langchain
import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging


# load environment variables from the .env file
load_dotenv()

# access the environment variables just liek you would with os.environ
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key = key, model_name = "gpt-3.5-turbo", temperature=0.7)

template = """
Text: {text}
you are a expert MCQ maker. Given the above text, its your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your reponse like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text","number","subject","tone","response_json"],
    template = template
)

quiz_chain = LLMChain(llm = llm, prompt = quiz_generation_prompt, output_key= "quiz", verbose = True)

template2 = """
you are an expert english grammerian and writer.given a multiple choice quiz for {subject} students.\
you need to evaluate the complexity of the question and give a complete analysis of the quiz only use at max 
50 words for complexity analysis.
if the quiz is not at per with the cognitive and analytical abilities of the students.\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits 
the student abilities
Quiz_MCQs:
{quiz}

check from an expert english writer of the above quiz:
"""

quiz_evaluation_template = PromptTemplate(
    input_variables=["subject","quiz"],
    template= template2
)

review_chain = LLMChain(llm = llm, prompt=quiz_evaluation_template, output_key="review", verbose=True)

generate_evaluate_chain = SequentialChain(chains=[quiz_chain,review_chain],
                                          input_variables=["text","number","subject","tone","response_json"],
                                          output_variables=["quiz","review"], verbose=True)

