from gtts import gTTS

def text_to_speech(text, filename, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    print(f"Generated audio file: {filename}")

# Example usage:
# text_to_speech("Hello Everyone", "reference_pronunciation.wav")
# text_to_speech("Hello Anyone", "student_pronunciation.wav")
# text_to_speech("Hello Everyone", "student_pronunciation.wav")

text_to_speech("Morning", "reference_pronunciation.wav")
text_to_speech("Good night", "student_pronunciation.wav")
