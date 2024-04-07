from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def example1():
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from langchain.chains import create_retrieval_chain
    from langchain.chains import create_history_aware_retriever

    # from langchain_core.prompts import MessagesPlaceholder
    # from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.tools.retriever import create_retriever_tool

    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()

    # llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )


def example_no_api():
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from langchain.chains import create_retrieval_chain
    from langchain.chains import create_history_aware_retriever

    # from langchain_core.prompts import MessagesPlaceholder
    # from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.tools.retriever import create_retriever_tool

    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()

    # llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "langsmith_search",
        "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
    )
    from langchain_community.tools.tavily_search import TavilySearchResults

    search = TavilySearchResults()
    tools = [retriever_tool, search]

    from langchain_openai import ChatOpenAI
    from langchain import hub
    from langchain.agents import create_openai_functions_agent
    from langchain.agents import AgentExecutor
    from langchain_core.messages import HumanMessage, AIMessage

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # You need to set OPENAI_API_KEY environment variable or pass it as argument `openai_api_key`.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    agent_executor.invoke({"input": "how can langsmith help with testing?"})
    agent_executor.invoke({"input": "what is the weather in SF?"})
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    result = agent_executor.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(result)


if __name__ == "__main__":
    # example1()
    example_no_api()
