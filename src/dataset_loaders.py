import json
from typing import Dict, List

import pandas as pd
from langchain.docstore.document import Document

from pyarrow.parquet import ParquetFile
import pyarrow as pa


def load_qasper_dataset(num_papers: int = 10) -> Dict[str, List[Document]]:
    QASPER_FILEPATH = "data/qasper_data/qasper-dev-v0.3.json"
    with open(QASPER_FILEPATH, "r") as f:
        corpus = json.load(f)

    paper_ids = sorted(list(corpus.keys()), reverse=True)
    if num_papers != -1:
        paper_ids = paper_ids[:num_papers]

    dataset = dict()
    for paper_id in paper_ids:
        paper = corpus[paper_id]

        metadata = {"title": paper["title"], "abstract": paper["abstract"]}
        paragraphs = []
        for section in paper["full_text"]:
            paragraphs.extend(section["paragraphs"])

        dataset[paper_id] = [
            Document(page_content=paragraph, metadata=metadata) for paragraph in paragraphs
        ]

    return dataset


def load_acl_dataset(num_papers: int = 10) -> Dict[str, Document]:
    ACL_74K_FILEPATH = "data/acl-publication-info.74k.parquet"

    pf = ParquetFile(ACL_74K_FILEPATH)
    head = next(pf.iter_batches(batch_size=num_papers * 5))
    df = pa.Table.from_batches([head]).to_pandas()
    df = df.loc[(df.abstract != "") & (df.full_text != "")]
    df = df.head(num_papers)

    dataset = dict()

    def add_row(row):
        metadata = {
            "title": row.title,
            "abstract": row.abstract,
            "corpus_paper_id": row.corpus_paper_id,
        }
        dataset[row.acl_id] = Document(page_content=row.full_text, metadata=metadata)
        return row

    df.apply(lambda row: add_row(row), axis=1)
    return dataset


if __name__ == "__main__":
    dataset = load_acl_dataset(100)
    for paper_id, paper_row in dataset.items():
        corpus_id = paper_row.metadata["corpus_paper_id"]
        print(paper_row.metadata)
        print(corpus_id)
