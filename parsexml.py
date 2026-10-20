"""

Parse patent XML file

Download from https://data.uspto.gov/bulkdata/datasets/appxml?fileDataFromDate=2025-10-9&fileDataToDate=2026-10-09 ?

The patent XML file has multiple XML records for each patent (ex. <?xml version="1.0" encoding="UTF-8"?>)

structure of patents:
    us-patent-application [lang="EN" dtd-version="v4.6 2022-02-17" file="US20260151258A1-20260604.XML" status="PRODUCTION" id="us-patent-application" country="US" date-produced="20260519" date-publ="20260604"]
        us-bibliographic-data-application [lang="EN" country="US"]
        abstract [id="abstract"]
        drawings [id="DRAWINGS"]
        description [id="description"]
        us-claim-statement -> What is claimed is:
        claims [id="claims"]

"""

import re
import xml.etree.ElementTree as ET
import json

with open("ipa260604.xml", encoding="utf-8") as f:
    content = f.read()

# Split right before each XML declaration, keeping the declaration with its chunk
chunks = re.split(r"(?=<\?xml\b)", content)
chunks = [c.strip() for c in chunks if c.strip()]

roots = []
for i, chunk in enumerate(chunks):
    try:
        roots.append(ET.fromstring(chunk.encode("utf-8")))
    except ET.ParseError as e:
        print(f"Document {i} failed to parse: {e}")

print(f"Parsed {len(roots)} documents")

patents_info = []

title = ""
abstract = ""
description = ""

def show_structure(elem, depth=0):
    global title, abstract, description

    if depth == 2:
        return
    
    attrs = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())
    text = (elem.text or "").strip()
    line = "  " * depth + elem.tag

    if elem.tag == "us-bibliographic-data-application":
            
        for child in elem:
            if child.tag == "invention-title":
                print(child.text)
                if child.text != None:
                    title = child.text

    if elem.tag == "abstract":
            
        for child in elem:
            print(child.text)
            if child.text != None:
                abstract += child.text

    if elem.tag == "description":
            
        for child in elem:
            print(child.text)
            if child.text != None:
                description += child.text + " "

    if title != "" and abstract != "" and description != "":
        #description = description.replace("")
        patents_info.append({ "title": title, "abstract": abstract, "description": description } )
        title = ""
        description = ""
        abstract = ""

    if attrs:
        line += f" [{attrs}]"
    if text:
        line += f" -> {text[:40]}"
    #print(line)
    for child in elem:
        show_structure(child, depth + 1)

roots = roots[:10000]

for i, root in enumerate(roots):
    print(f"\n=== Document {i} ===")
    show_structure(root)

with open("out.jsonl", "w", encoding="utf-8") as f:
    for rec in patents_info:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

