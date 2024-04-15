from VocabExtractor import VocabExtractor

if __name__ == '__main__':
    vocab_extractor = VocabExtractor("vocab.pdf")
    extracted_data = vocab_extractor.extract_from_pdf()

    print(extracted_data)