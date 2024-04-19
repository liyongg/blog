import pandas as pd
from pypdf import PdfReader

class VocabExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def validate_lines(self, lines):
        return [line for line in lines if line and line[0].isdigit()]

    def remove_overflow_lines(self, lines):
        res = [lines[0]]
        for current_item, next_item in zip(lines, lines[1:]):
            if next_item[0].isdigit():
                res.append(next_item)
            else:
                res[-1] += next_item
        return res

    def trim_index(self, lines):
        no_index_lines = [line[line.find(' '):] for line in lines]
        return [line.strip() for line in no_index_lines]

    def lines_to_df(self, lines):
        split_lines = [line.split("  ") for line in lines]
        words = [line[0] for line in split_lines]
        meanings = [''.join(line[1:]).strip() for line in split_lines]
        return pd.DataFrame.from_dict({"Words": words, "Meanings": meanings})

    def pipeline_lines(self, text):
        lines = text.splitlines()
        page_lines = self.validate_lines(lines)
        no_overflow_lines = self.remove_overflow_lines(page_lines)
        no_index_lines = self.trim_index(no_overflow_lines)
        return self.lines_to_df(no_index_lines)

    def extract_from_pdf(self):
        reader = PdfReader(self.pdf_path)
        pages = reader.pages

        res = []

        for page in pages:
            page_text = page.extract_text(extraction_mode="layout")
            res.append(self.pipeline_lines(page_text))

        df = pd.concat(res, ignore_index=True)
        df = df[df["Words"] != "Derde Ronde Nederlands voor buitenlanders"].reset_index(drop=True)

        return df
