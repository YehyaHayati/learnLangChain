from langchain.agents import create_react_agent, AgentExecutor
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_tavily import TavilySearch
from langchain import hub
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import re
load_dotenv()

class YoutubeTranscriptToolInput(BaseModel):
    url: str = Field(required=True, description="The URL of the YouTube video to get the Transcript of.")

class YoutubeTranscriptTool(BaseTool):
    name: str = 'YouTube Transcript Tool'
    description: str = 'A tool used to get the Transcript of a YouTube video.'
    args_schema: Type[BaseModel] = YoutubeTranscriptToolInput

    def _run(self, URL: str):
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", URL)
        return YouTubeTranscriptApi().fetch(video_id=match, languages=["en"])

tavily_search_tool = TavilySearch(
    max_results=1,
    topic="general",
)
# result = tavily_search_tool.invoke({"query": "Hotest AI news on youtube."})
# for res in result['results']:
#     print('_' * 50)
#     print(res['url'])

youtube_transcript_tool = YoutubeTranscriptTool()
    
prompt = hub.pull("hwchase17/react")
tools = [tavily_search_tool, youtube_transcript_tool]
agent = create_react_agent(llm=ChatOpenAI(), tools=tools, prompt=prompt)
agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_exec.invoke({"input":"Find the youtube video with the latest AI news and summarize it for me."})
print(result)