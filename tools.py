from typing import Type
from langchain_community.tools import CopyFileTool, DeleteFileTool, DuckDuckGoSearchRun, tool, StructuredTool, BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

# Built-in Tools
# print(DuckDuckGoSearchRun().invoke("Latest news about AI."))
# CopyFileTool().invoke({'source_path':'./RAG.py', 'destination_path':'./P02.py'})
# DeleteFileTool().invoke('./P02.py')

#Custom Tools
#________________________________________________________
# Using the @tool decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies 2 numbers"""
    return a*b

result = multiply.invoke({"a":3, "b":4})
print(result)

#________________________________________________________
# Using StructuredTool
def multiply_func(a: int, b: int) -> int:
    return a*b

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="Multiply",
    description="Multiply 2 numbers",
    args_schema=MultiplyInput
)
result = multiply_tool.invoke({"a":2, "b":7})
print(result)

#________________________________________________________
# Using BaseTool => Used to make the above 2 methods
class MultiplytTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two number"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b
    
multiply_tool2 = MultiplytTool()
result = multiply_tool2.invoke({'a': 2, 'b':8})
print(result)

#________________________________________________________

# Toolkits -> Collection of related tools that serve a common functionality
@tool
def add(a:int, b:int) -> int:
    """Adds two numbers"""
    return a + b

class MathToolKit:
    def get_tools(self):
        return [add, multiply]
    
toolkit = MathToolKit()
tools = toolkit.get_tools()
for tool in tools:
    print(tool.name, "=>", tool.description)
