from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition
from agent import AgentState, ai_assistant, generate, grade_documents, rewriter
from tools import retrieve


graph = StateGraph(AgentState)

graph.add_node("AI_assistant", ai_assistant)
graph.add_node("Vector_retriever", retrieve)
graph.add_node("Output_generator", generate)
graph.add_node("Query_rewriter", rewriter)

graph.add_edge(START, "AI_assistant")
graph.add_conditional_edges(
    "AI_assistant",
    tools_condition,
    {
        "tools": "Vector_retriever", END:END,
    }
)

graph.add_conditional_edges(
    "Vector_retriever",
    grade_documents,
    {
        "generator": "Output_generator",
        "rewriter": "Query_rewriter",
    }
)

graph.add_edge("Query_rewriter", "Output_generator")
graph.add_edge("Output_generator", END)

app = graph.compile()
state = {"messages":["Tell me about llama"]}

output = app.invoke(state)

print("<--- Final Output --->")
#print(state['messages'][-1].content)
print("<--- Final Output --->"*5)
print(output['messages'][-1].content[0]['text'])