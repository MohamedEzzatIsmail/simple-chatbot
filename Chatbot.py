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
    return responses[most_similar_index] if cosine_similarities[
                                                most_similar_index] > 0 else "I'm sorry, I didn't understand that. Can you please rephrase?"


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
    nltk.download('punkt')  # Download necessary data for nltk
    run_chatbot()