from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def simple_retrieval():
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()

    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
    )

    llm = ChatOpenAI()
    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])


def conversation_retrieval_example():
    from langchain_openai import OpenAIEmbeddings
    from langchain.chains import create_retrieval_chain
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains.combine_documents import create_stuff_documents_chain

    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()

    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    # First we need a prompt that we can pass into an LLM to generate this search query

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    # We can test this out by passing in an instance where the user asks a follow-up question.
    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = [
        HumanMessage(content="Can LangSmith help test my LLM applications?"),
        AIMessage(content="Yes!"),
    ]
    retriever_chain.invoke({"chat_history": chat_history, "input": "Tell me how"})

    # Now that we have this new retriever, we can create a new chain to continue the conversation with these retrieved documents in mind.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    result = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    print(result)
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    # simple_retrieval()
    conversation_retrieval_example()
