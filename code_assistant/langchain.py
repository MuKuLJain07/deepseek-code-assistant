from decouple import config
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

HUGGINGFACEHUB_API_TOKEN = config('HUGGINGFACEHUB_API_TOKEN')
model_repo_id = "deepseek-ai/deepseek-coder-1.3b-instruct"

def askProblem(query):
    llm = HuggingFaceEndpoint(repo_id=model_repo_id, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

    chat = ChatHuggingFace(llm=llm, verbose=True)

    systemMessagePrompt = SystemMessagePromptTemplate.from_template("You are a coding assistant with expertise in multiple programming languages and technologies. Your task is to help users with coding-related queries, providing solutions, explanations, and guidance in a clear, concise, and efficient manner. Always ensure that your code aligns with best practices, is efficient, and comnsiders edge cases. Introduce yourself as a coding assistant.")

    humanMessagePromp = HumanMessagePromptTemplate.from_template("{query}")

    chatPrompt = ChatPromptTemplate.from_messages([systemMessagePrompt, humanMessagePromp])
    formattedChatPrompt = chatPrompt.format_messages(problem=query)
    response = chat.invoke(formattedChatPrompt)
    return response.content
