import time
from VocabExtractor import VocabExtractor
from transformers import pipeline

pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M", device="cuda:0")

def translate(tokens, src_lang="nld_Latn", tgt_lang="zho_Hans"):
    start = time.time()
    translation = pipe(tokens, src_lang=src_lang, tgt_lang=tgt_lang)
    end = time.time()
    return {"translation": translation, "time": end-start}



if __name__ == '__main__':
    vocab_extractor = VocabExtractor("vocab.pdf")
    df = vocab_extractor.extract_from_pdf()
    df["Words_trimmed"] = df["Words"].replace(regex=r'\s\((de|het)\)', value='')
    df["Meanings_trimmed"] = df["Meanings"] \
        .replace(regex=r';\s*[\w\s]+-[\w\s]+-[\w\s]+', value='') \
        .replace(regex=r'\((de|het)\)', value='') \
        .str.strip()
    
    words = df["Words_trimmed"].tolist()
    meanings = df["Meanings_trimmed"].tolist()
    
    title = translate("Translating PDF content using a Large Language Model", src_lang="eng_Latn")
    title["translation"]["translation_text"]
    
    translations_words_nllb = translate(words)
    translations_meanings_nllb = translate(meanings)
    
    df["CN_Words_NLLB"] = [item["translation_text"] for item in translations_words_nllb["translation"]]
    df["CN_Meanings_NLLB"] = [item["translation_text"] for item in translations_meanings_nllb["translation"]]
    df.to_csv("vocab.csv", sep=";", index=False, encoding="utf-8-sig")
    df