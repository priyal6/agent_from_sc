from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
import uuid
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_tool
load_dotenv()
os.environ.get("OPENAI_API_KEY")

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

system_prompt = f"""
You are a research assistant.

You MUST use the search tool at least once before answering.
You must include at least 1 source (URL or page title) that came from the tool output.
In tools_used, include the names of any tools you called.
After producing the final JSON, call the save_text_to_file tool
with the JSON string as input.


{parser.get_format_instructions()}
""".strip()

tools = [search_tool, wiki_tool, save_tool]
agent = create_agent(
    model = llm,
    tools= tools,
    system_prompt=system_prompt
)

raw_response =  agent.invoke({
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
})
#print(raw_response)


text = raw_response["messages"][-1].content  # last message is AIMessage
structured_response = parser.parse(text)
print(structured_response)

