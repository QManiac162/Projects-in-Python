import openai
import os

# loading the api key from an environment variable
openai.api_key = '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'

def chatbot():
    # Create a list to store all the messages for context
    messages = [{"role": "system", "content": "You are a English assistant."},]
    
    while True:
        # prompt user to input
        message = input("User: ")
        
        # exit program if user inputs "quit"
        if message.lower() == "quit":
            break
        
        # add each new message to the list
        messages.append({"role": "user", "content": message})
        
        # request gpt-3.5-turbo for chat completion
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = messages
        )
        
        # print the response and add it to the message list
        chat_message = response['choices'][0]['message']['content']
        print(f"chatter: {chat_message}")
        messages.append({"role": "assistant", "content": chat_message})
        
if __name__ == "__main__":
    print("Start chatting with chatter! \n(type 'quit' to stop)!")
    chatbot()