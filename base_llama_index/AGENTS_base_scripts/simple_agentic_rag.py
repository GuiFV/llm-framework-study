from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool

load_dotenv()

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# Settings.llm = Ollama(model="mixtral:8x7b", request_timeout=360.0)
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

# llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = Ollama(model="mixtral:8x7b", request_timeout=120.0)
# agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)
# response = agent.chat("What is 20+(2*4)? Use a tool to calculate every step.")
# print(response)

# load document
documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# USE LOADED DOCUMENTS AS TOOL
budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget.",
)

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, budget_tool], verbose=True
)

response = agent.chat("What is the total amount of the 2023 Canadian federal budget multiplied by 3? Go step by step, "
                      "using a tool to do any math.")

print(response)
