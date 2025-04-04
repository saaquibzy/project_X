import speech_recognition as sr
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed
engine.setProperty('volume', 1)  # Volume

def speak(text):
    """Converts text to speech."""
    print(f"Chatbot: {text}")
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    """Listens to user's voice and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening... 🎤")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        try:
            audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds
            user_input = recognizer.recognize_google(audio)  # Convert speech to text
            print(f"You: {user_input}")
            return user_input.lower()
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that. Can you repeat?")
            return None
        except sr.RequestError:
            speak("There is a problem with speech recognition.")
            return None

def chatbot_response(user_input):
    """Generates chatbot responses based on input."""
    if user_input is None:
        return "I didn't understand. Please try again."

    elif "hello" in user_input:
        return "Hello! How can I assist you today?"

    elif "how are you" in user_input:
        return "I'm just a chatbot, but I'm doing great! How about you?"

    elif "your name" in user_input:
        return "I'm your AI assistant."

    elif "exit" in user_input or "bye" in user_input:
        return "Goodbye! Have a great day!"

    else:
        return "I'm not sure how to respond to that."

# Main chatbot loop
if __name__ == "__main__":
    speak("Hello! I am your chatbot. Say 'exit' to stop.")

    while True:
        user_input = recognize_speech()
        if user_input:
            response = chatbot_response(user_input)
            speak(response)

            if "bye" in user_input or "exit" in user_input:
                break  # Exit loop if user says 'bye' or 'exit'

    speak("Chatbot shutting down.")



hi
