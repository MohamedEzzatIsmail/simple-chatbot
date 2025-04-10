# simple-chatbot
Overview
This project implements a multifunctional chatbot that can:

Respond to user queries based on a predefined dataset.
Tell the current time.
Self-learn from user interactions by updating its knowledge base.
The chatbot uses Natural Language Processing (NLP) techniques to understand user input and provide relevant responses.

Features
Self-Learning: The chatbot learns from user feedback. If a user indicates that a response was not helpful, they can provide a better response, which is saved for future interactions.

Tell the Time: The chatbot can provide the current time when prompted.

Installation
Prerequisites
Python 3.x
An internet connection (for downloading NLTK data)
Required Libraries
You need to install the following Python libraries:


pip install nltk scikit-learn
Download NLTK Data
The chatbot uses the NLTK library for natural language processing. You need to download the necessary data:


import nltk
nltk.download('punkt')
Usage
Create the Dataset: Create a file named dialogs.txt in the same directory as your Python script. Populate it with question-response pairs in the following format:


question1    response1
question2    response2
Example content for dialogs.txt:


hi    I'm fine. how about yourself?
what is your name?    I am your chatbot.
how are you?    I'm just a program, but thanks for asking.
Run the Chatbot: Execute the Python script. You can interact with the chatbot in the console.


Interacting with the Chatbot: Type your queries in the console. To exit the chatbot, type bye.

Code Structure
Main Components
load_responses(file_path): Loads question-response pairs from the specified text file.

get_response(user_input): Processes user input to determine the appropriate response. It checks for specific commands (like asking for the time) and uses TF-IDF and cosine similarity to find the best match from the dataset.

run_chatbot(): The main loop that handles user interaction, collects feedback, and updates the dataset.

Example Code
Here is the main code for the chatbot:


import nltk
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load responses from a text file
def load_responses(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            question, response = line.strip().split('\t')
            pairs.append((question.strip(), response.strip()))
    return pairs

# Load responses from the local .txt file
responses_file = 'dialogs.txt'
pairs = load_responses(responses_file)

# Prepare the questions for TF-IDF
questions = [question for question, response in pairs]
responses = [response for question, response in pairs]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def get_response(user_input):
    # Check for specific commands
    if "time" in user_input.lower():
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    
    # Transform the user input to the same TF-IDF space
    user_input_tfidf = vectorizer.transform([user_input])
    
    # Calculate cosine similarity between user input and questions
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()
    
    # Find the index of the most similar question
    most_similar_index = cosine_similarities.argmax()
    
    # Return the corresponding response
    return responses[most_similar_index] if cosine_similarities[most_similar_index] > 0 else "I'm sorry, I didn't understand that. Can you please rephrase?"

def run_chatbot():
    print("Hello! I'm your multifunctional chatbot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print("Chatbot:", response)
        
        # Collect feedback for self-learning
        feedback = input("Was this response helpful? (yes/no): ")
        if feedback.lower() == 'no':
            new_response = input("Please provide a better response: ")
            with open(responses_file, 'a') as file:
                file.write(f"{user_input}\t{new_response}\n")
            print("Thank you for your feedback!")

if __name__ == "__main__":
    nltk.download('punkt')
    run_chatbot()

    
Conclusion
This chatbot project provides a foundation for building a multifunctional conversational agent. You can expand its capabilities by adding more features, improving the dataset, and integrating additional APIs for enhanced functionality.

If you have any questions or need further assistance, feel free to reach out!
