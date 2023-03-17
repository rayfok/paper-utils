import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import jsonlines
import openai
import requests
from tqdm import tqdm

from dataset_loaders import load_acl_dataset
from my_secrets import openai_api_key, openai_organization

openai.organization = openai_organization
openai.api_key = openai_api_key


S2ORC_BASE_API_URL = "http://s2orc-api.prod.s2.allenai.org/paper"
CACHE_PATH = Path.home() / ".cache" / "paper-utils"
OUTPUT_PATH = Path("output")


def extract_contributions(full_text: str, model: str = Literal["gpt3", "gpt4"]):
    intro = ""
    for section, text in full_text.items():
        if "introduction" in section.lower():
            intro = " ".join(text)
            break

    if not intro:
        return []

    if "contribu" not in intro.lower():
        return []

    COMPLETIONS_API_PARAMS = {"temperature": 0.0, "max_tokens": 500, "model": "text-davinci-003"}
    CHAT_COMPLETIONS_API_PARAMS = {
        "temperature": 0.0,
        "max_tokens": 300,
        "model": "gpt-3.5-turbo",
    }
    prompt = f"""
        Extract author-defined contributions from the following introduction. Contributions should be listed verbatim, copied exactly as they appear in the introduction. Author-defined contributions will generally start with a phrase such as "This paper contributes" or "We make the following contributions".

        If there are no contributions explicitly stated by the authors, respond only with 'No contributions'.

        Introduction: {intro}

        Contributions:
        -
    """

    try:
        if model == "gpt4":
            response = openai.ChatCompletion.create(
                **CHAT_COMPLETIONS_API_PARAMS,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an assistant designed to identify contributions author claim within a research paper. The paper might not contain any contributions explicitly stated by the authors; respond with 'No contributions' if that is the case.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            result = response.choices[0].message.content.strip()

        elif model == "gpt3":
            response = openai.Completion.create(prompt=prompt, **COMPLETIONS_API_PARAMS)
            result = response.choices[0].text.strip()
    except Exception as e:
        result = "FAILED"
        print("FAILED:", e)

    if result == "No contributions":
        return []

    result = result.split("\n")
    result = [re.sub(r"[^A-Za-z0-9 ]+", "", c).strip() for c in result]
    return result


def load_paper(corpus_id: str):
    # if (CACHE_PATH / "s2orc_data" / corpus_id).is_file():
    #     return json.load()
    res = requests.get(f"{S2ORC_BASE_API_URL}/{corpus_id}")
    data = res.json()
    if res.status_code == 404:
        return None
    try:
        metadata = data["metadata"]
        grobid = data["content"]["grobid"]
        full_text = grobid["contents"]
        annotations = grobid["annotations"]
        section_headers = annotations["section_header"]
        paragraphs = annotations["paragraph"]
    except KeyError:
        return None

    sections_visited: set[tuple[str, str]] = set()
    parsed_sections = []
    for section_header in section_headers:
        # if (
        #     "attributes" not in section_header
        # ):  # Skip non-section headers (e.g., figure captions, table headers)
        #     continue

        start, end = section_header["start"], section_header["end"]
        if (start, end) in sections_visited:
            continue

        header_text = full_text[start:end]
        sections_visited.add((start, end))
        parsed_sections.append((header_text, start, end))

    # Create ranges of spans corresponding to text in each section
    section_ranges = []
    for i in range(len(parsed_sections) - 1):
        header_text = parsed_sections[i][0]
        end1 = parsed_sections[i][2]
        start2 = parsed_sections[i + 1][1]
        section_ranges.append((header_text, end1, start2))
    header_text, start, end = parsed_sections[-1]
    section_ranges.append((header_text, end, None))

    paragraphs_visited: set[tuple[str, str]] = set()
    full_text_by_sections = defaultdict(list)
    for header_text, section_start, section_end in section_ranges:
        for paragraph in paragraphs:
            start, end = paragraph["start"], paragraph["end"]
            if (start, end) in paragraphs_visited:
                continue
            if start < section_start or (section_end and end > section_end):
                continue

            paragraph_text = full_text[start:end]
            paragraphs_visited.add((start, end))
            full_text_by_sections[header_text].append(paragraph_text)

    paper = {"metadata": metadata, "full_text": full_text_by_sections}
    return paper


def main():
    # Load previously extracted contributions
    processed_corpus_ids = set()
    with jsonlines.open(OUTPUT_PATH / "acl_paper_contributions_v1.jsonl") as reader:
        for row in reader:
            processed_corpus_ids.add(row["id"])

    # Get corpus ids for paper we want to extract contributions from
    paper_ids = []
    dataset = load_acl_dataset(3000)
    for paper_id, paper_row in dataset.items():
        corpus_id = paper_row.metadata["corpus_paper_id"]
        paper_ids.append(corpus_id)

    # Extract contributions and save to output file
    with jsonlines.open(OUTPUT_PATH / "acl_paper_contributions_v1.jsonl", mode="a") as writer:
        for paper_id in tqdm(paper_ids):
            if paper_id in processed_corpus_ids:
                print(f"{paper_id} already processed")
                continue

            paper = load_paper(paper_id)
            if not paper:
                print(f"{paper_id} failed (not in s2orc or missing data)")
                continue
            contributions = extract_contributions(paper["full_text"], model="gpt4")
            writer.write(
                {
                    "id": paper_id,
                    "title": paper["metadata"].get("title", ""),
                    "contributions": contributions,
                }
            )


if __name__ == "__main__":
    main()
