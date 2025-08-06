from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Type
from dotenv import load_dotenv
load_dotenv()

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number used in the product.")
    b: int = Field(required=True, description="The second number used in the product.")

class MultiplyTool(BaseTool):
    name: str = "Mulitply"
    description: str = "A tool used to multiply two numbers."
    args_schema: Type[BaseModel] = MultiplyInput
    def _run(self, a: int, b: int) -> int:
        return a * b
    
tool = MultiplyTool()
llm = ChatOpenAI().bind_tools([tool])
result = llm.invoke("What is 30*10?")
print(result.tool_calls[0])

tool_result = tool.invoke(result.tool_calls[0])
print(tool_result)

# If a tool's parameter is another tool's output, we can use 
# from langchain_core.tools import InjectedToolArg
# within an Annotated to mark a dependency