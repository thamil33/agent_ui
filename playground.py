from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.models.lmstudio import LMStudio
from agno.playground import Playground
from agno.memory.v2.memory import Memory
from agno.tools.reasoning import ReasoningTools
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os
from agno.memory.v2.db.sqlite import SqliteMemoryDb

load_dotenv()

lmstudio_base_url =  os.getenv("LM_STUDIO_BASE_URL")
OR_api_key = os.getenv("OPENROUTER_API_KEY")

lmstudio_model=LMStudio(id="llama-3.2-8x3b-abliterated",base_url=lmstudio_base_url)
or_gemini_model = OpenRouter(id="deepseek/deepseek-chat-v3-0324:free", api_key=OR_api_key)
or_mistral_model = OpenRouter(id="mistralai/mistral-small-3.2-24b-instruct:free", api_key=OR_api_key)

memory = Memory(
    model=or_mistral_model,
    db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/agents.db"),
    delete_memories=True,
    clear_memories=True,
)


# Agent using OpenRouter
openrouter_agent = Agent(
    name="OpenRouter Agent",
    model= or_gemini_model,
    tools=[
        ReasoningTools(add_instructions=True),
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True),
    ],
    instructions=[
        "Use tables to display data.",
        "Include sources in your response.",
        "Only include the report in your response. No other text.",
    ],
    markdown=True,
    add_history_to_messages=True,
    num_history_responses=5,
)


# Agent using LM Studio (make sure you have a model running locally)
lmstudio_agent = Agent(
    name="LM Studio Agent", 
    model=or_gemini_model,  
    tools=[ ReasoningTools(add_instructions=True)],
    instructions=["Use your reasoning only when confronted with a complex problem.", 
                  "You are a general chat AI assistant.",
                  ],
    memory=memory,
    markdown=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
)


# Create playground with both agents
playground = Playground(agents=[openrouter_agent, lmstudio_agent])
app = playground.get_app()

if __name__ == "__main__":
    playground.serve("playground:app", reload=True)