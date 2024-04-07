from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def example1():
    llm = ChatOpenAI()
    result = llm.invoke("how can langsmith help with testing?")
    print(result)


def example2():
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are world class technical documentation writer."),
            ("user", "{input}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"input": "how can langsmith help with testing?"})
    print(result)


def example3():
    """The output of a ChatModel (and therefore, of this chain) is a message.
    However, it's often much more convenient to work with strings.
    Let's add a simple output parser to convert the chat message to a string."""
    from langchain_core.output_parsers import StrOutputParser

    output_parser = StrOutputParser()

    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are world class technical documentation writer."),
            ("user", "{input}"),
        ]
    )
    chain = prompt | llm | output_parser
    result = chain.invoke({"input": "how can langsmith help with testing?"})
    print(result)


if __name__ == "__main__":
    # example1()
    # example2()
    example3()
