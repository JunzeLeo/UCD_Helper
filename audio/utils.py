import os

import langchain
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.memory import ChatMessageHistory
from langchain_community.cache import SQLiteCache
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.history import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
)
from langchain_openai import ChatOpenAI

set_llm_cache(
    SQLiteCache(database_path=".langchain_cache.db")
)  # setup cache for the LLM

langchain.debug = False  # set verbose mode to True to show more execution details

load_dotenv()

# # Use Azure's models
# langchain_llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("MY_AZURE_ENDPOINT"),
#     api_key=os.getenv("MY_AZURE_API_KEY"),
#     api_version=os.getenv("MY_AZURE_API_VERSION"),
#     azure_deployment=os.getenv("MY_AZURE_DEPLOYMENT_NAME"),
#     verbose=True,
# )

# Or use OpenAI's model
langchain_llm = ChatOpenAI(
    base_url=os.getenv("MY_OPENAI_API_BASE"),
    api_key=os.getenv("MY_OPENAI_API_KEY"),
    model="gpt-4o",
    verbose=True,
)


class UserCenteredDesignNotes(BaseModel):
    hear: str = Field(
        description="What the user hears in their environment, including influences from friends, family, colleagues, and media.",
        example="A user might hear a friend recommending a new app or a colleague discussing a recent trend in technology.",
    )
    see: str = Field(
        description="What the user sees in their environment, including visual aspects like physical surroundings, advertisements, and the behavior of others.",
        example="A user might see a billboard advertising a new product or observe how others are using a particular service.",
    )
    say: str = Field(
        description="What the user says out loud in conversations, including their opinions, feedback, and verbal expressions about their experiences and challenges.",
        example="A user might say they find a particular interface difficult to navigate or express excitement about a new feature.",
    )
    do: str = Field(
        description="The user's actions and behaviors, including their habits, routines, and interactions with products or services.",
        example="A user might frequently check their phone for notifications or use a specific app regularly.",
    )
    think: str = Field(
        description="The user's thoughts and beliefs, including their motivations, goals, and what occupies their mind, such as worries and aspirations.",
        example="A user might think about how to improve their productivity or worry about data privacy when using a new app.",
    )
    feel: str = Field(
        description="The user's emotions and feelings, including their emotional responses to experiences, which can range from frustration and anxiety to joy and satisfaction.",
        example="A user might feel frustrated with a slow-loading website or feel delighted when they receive excellent customer service.",
    )


notes_parser = JsonOutputParser(UserCenteredDesignNotes)

system_message = "The user-centered design notes provide insights into the user's environment, actions, thoughts, and emotions. These notes help understand the user's perspective and tailor the conversation to their needs and preferences. Please summarize the user-centered design notes from the user's speaking transcript in a co-design workshop, and summarize the key points in the chatbot conversation. Based on the user-centered design notes, the chatbot can provide personalized recommendations and support to enhance the user experience. Please provide the user-centered design notes in the following format: {format_instructions}."

# baseline_chatbot_prompt = ChatPromptTemplate.from_messages(
#     [
#         system_message,
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#     ]
# )

baseline_chatbot_prompt = ChatPromptTemplate(
    messages=[
        system_message,
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ],
    partial_variables={"format_instructions": notes_parser.get_format_instructions()},
)


baseline_chatbot_chain = baseline_chatbot_prompt | langchain_llm

baseline_chatbot_history = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    baseline_chatbot_chain,
    lambda session_id: baseline_chatbot_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def summarize_messages(chain_input):
    stored_messages = baseline_chatbot_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | langchain_llm
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    baseline_chatbot_history.clear()
    baseline_chatbot_history.add_message(summary_message)
    return True


chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_message_history
)
