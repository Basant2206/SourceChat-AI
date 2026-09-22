from retriever import retriever
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import ToolNode
print("Loading retriever...")
retriever = retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_",
    "Search and return information related to query. You are a specialized assistant. Use the 'retriever_tool' **only** when the query explicitly relates to LangChain blog data. For all other queries, respond directly without using any tool. For simple queries like 'hi', 'hello', or 'how are you', provide a normal response.",   
)


retrieve = ToolNode([retriever_tool])

tools = [retriever_tool]
