from langchain_community.retrievers import TavilySearchAPIRetriever

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def simple_search():
    retriever = TavilySearchAPIRetriever(k=3)
    result = retriever.invoke("what year was breath of the wild released?")
    print(result)


def chaining():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI

    retriever = TavilySearchAPIRetriever(k=3)
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.

    Context: {context}

    Question: {question}"""
    )
    chain = (
        RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever)
        | prompt
        | ChatOpenAI(model="gpt-4-1106-preview")
        | StrOutputParser()
    )
    result = chain.invoke(
        {"question": "how many units did bretch of the wild sell in 2020"}
    )
    print(result)
