from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from dotenv import load_dotenv
# to use llama llm locally
from llama_index.llms.ollama import Ollama

# load open ai api key
load_dotenv()

# not obligatory. setup of llm to be used
# Settings.llm = OpenAI(temperature=0.2, model="gpt-4")

# run it
# response = OpenAI().complete("Paul Graham is ")

# OR: to use locally
Settings.llm = Ollama(model="llama3.2", request_timeout=60.0)
response = Settings.llm.complete("Paul Graham is ")

# get the answer back
print(response)