from setuptools import find_packages,setup

setup(
    name = "mcqgenerator",
    version= "0.0.1",
    author= "sam_pawar",
    author_email= "sampawar2577@gmail.com",
    install_reqires = ["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages = find_packages()
)

