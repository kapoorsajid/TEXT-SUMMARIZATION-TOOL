import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text(input_text, max_length=130, min_length=30, do_sample=False):
    # Encode the input text
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, do_sample=do_sample)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    # Example input text (you can replace this with any lengthy article)
    input_text = """
    Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to enable computers to understand, interpret, and generate human language in a way that is valuable. NLP combines computational linguistics with machine learning and deep learning techniques to process and analyze large amounts of natural language data. Applications of NLP include sentiment analysis, language translation, chatbots, and information extraction, among others. As technology continues to evolve, the capabilities of NLP are expanding, leading to more sophisticated and effective applications in various industries.
    """

    # Generate summary
    summary = summarize_text(input_text)
    
    # Print the original text and the summary
    print("Original Text:\n", input_text)
    print("\nSummary:\n", summary)