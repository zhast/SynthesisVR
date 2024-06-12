import whisper
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio(audio_file):
    logging.info(f"Transcribing audio file: {audio_file}")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcribed_text = result["text"]
    logging.info(f"Transcription completed: {transcribed_text}")
    return transcribed_text

def generate_prompt(transcribed_text):
    logging.info("Generating prompt using OLLaMA...")
    prompt = f"The following text was transcribed from an audio file: '{transcribed_text}'. Based on this text, generate a concise one-sentence prompt that can be used as input for a text-to-image diffusion model like Stable Diffusion. Focus on the essential details needed to create the desired image, such as the main subject, key visual elements, and overall style or mood. Provide only the generated prompt without any additional explanations or commentary."
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    logging.info("Sending request to OLLaMA API...")
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        logging.info("Request successful. Processing response...")
        generated_text = ""
        for line in response.text.strip().split("\n"):
            chunk = json.loads(line)
            generated_text += chunk["response"]
        generated_prompt = generated_text.strip()
        logging.info(f"Generated prompt: {generated_prompt}")
        return generated_prompt
    else:
        logging.error(f"Request failed with status code: {response.status_code}")
        return None

def main():
    audio_file = "audio.m4a"
    transcribed_text = transcribe_audio(audio_file)
    generated_prompt = generate_prompt(transcribed_text)
    
    if generated_prompt:
        logging.info("Script execution completed successfully.")
    else:
        logging.error("Script execution failed.")

if __name__ == "__main__":
    main()