from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


print("=== Local Product Assistant Bot (FREE) ===")
print("Type 'exit' to quit")

# Load local LLM
llm = ChatOllama(model="llama3")

# System prompt
system_msg = SystemMessage(
    content="You are a helpful assistant for answering questions about home appliance usage such as trimmers, air fryers, washing machines, etc."
)

history = [system_msg]

while True:
    user = input("\nYou: ")

    if user.lower() == "exit":
        print("Bot: Bye! 👋")
        break

    history.append(HumanMessage(content=user))

    response = llm.invoke(history)

    history.append(response)

    print("Bot:", response.content)
