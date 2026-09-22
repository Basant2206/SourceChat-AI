from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from tools import tools
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
def load_params(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

params = load_params('params.yaml')
llm_model_name = params['model']['llm']
llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=0.2, max_output_tokens=512)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

def ai_assistant(state:AgentState):
    print("<---Call Agent --->")
    messages = state['messages']
    if len(messages) > 1:
        last_message = messages[-1]
        question = last_message.content
        prompt = PromptTemplate(
            template="""You are a helpful assistant whatever question has been asked to find out that in the given question and answer.
            Here is the question:{question}
            """,
            input_variables=["question"]
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})
        return{"messages": [response]}
    else:
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)
        return {'messages':[response]}

class grade(BaseModel):
    binary_score:str = Field(description="Relevance score 'yes' or 'no'")

def grade_documents(state:AgentState)-> Literal["Output_Generator","Query_Rewriter"]:
    llm_with_structure_op = llm.with_structured_output(grade)
    prompt = PromptTemplate(
        template = """You are a grader deciding, if a document is relevant to a user's question.
        Here is the document: {context}
        Here is the question asked by user: {question}
        If the document talks about or contains information related to the user's question, mark it as relevant.
        Give a 'yes' or 'no' answer to show if the document is relavant to the question.      
        """,
        input_variables=["context","question"]
        )
    chain = prompt | llm_with_structure_op
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context":docs})
    score = scored_result.binary_score
    if score == "yes":
        print("<---Decision: docs relavant --->")
        return "generator"
    else:
        print("<---Decision: docs not relevant --->")
        return "rewriter"

def generate(state:AgentState):
    print("<--- Generator --->")
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the question using only the provided context.
        
        Context: {context}
        Question:{question}
        """
           )

    rag_chain = prompt | llm
    response = rag_chain.invoke({"context":docs, "question":question})
    print(f"This response: {response}")

    return {'messages':[response]}

def rewriter(state:AgentState):
    print("<--- Transform Query --->")
    messages = state["messages"]
    question = messages[0].content
    message = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent or meaning.
            Here is the initial question: {question}
            Formulate and improve question:def""")
           ]
    response = llm.invoke(message)
    return {"messages": [response]}

