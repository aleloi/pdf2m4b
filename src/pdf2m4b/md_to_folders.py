#!/usr/bin/env python3
"""
md_to_folders.py

Parses a Markdown file (produced by pymupdf4llm) and converts it into a
hierarchical folder structure. Each heading creates a folder and its
associated content is written into a "00.md" file inside that folder.
"""

import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from .logger_config import logger

@dataclass
class Node:
    level: int
    title: str
    content: List[str]
    children: List["Node"] = field(default_factory=list)

def parse_markdown(file_path: Path) -> Node:
    heading_re = re.compile(r"^(#+)\s+(.*)")
    root = Node(level=0, title="root", content=[])
    stack: List[Node] = [root]
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                m = heading_re.match(line)
                if m:
                    hashes, title = m.groups()
                    level = len(hashes)
                    node = Node(level=level, title=title.strip(), content=[])
                    while stack and stack[-1].level >= level:
                        stack.pop()
                    stack[-1].children.append(node)
                    stack.append(node)
                else:
                    stack[-1].content.append(line)
    except Exception as e:
        logger.error("Error reading markdown file", file=str(file_path), error=str(e))
        raise
    return root

def sanitize(title: str) -> str:
    title = title.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", title)

def process_children(parent: Node, out_folder: Path) -> None:
    for idx, child in enumerate(parent.children, start=1):
        folder_name = f"{idx:02d}_{sanitize(child.title)}"
        child_folder = out_folder / folder_name
        child_folder.mkdir(exist_ok=True)
        if child.content:
            content_path = child_folder / "00.md"
            try:
                with content_path.open("w", encoding="utf-8") as f:
                    f.write("\n".join(child.content))
            except Exception as e:
                logger.error("Error writing markdown content", file=str(content_path), error=str(e))
        process_children(child, child_folder)

def convert_md(input_md: Path, output_folder: Path) -> None:
    try:
        output_folder.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.error("Error creating output folder", folder=str(output_folder), error=str(e))
        raise

    root = parse_markdown(input_md)
    if root.content:
        preamble_file = output_folder / "00_preamble.md"
        try:
            with preamble_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(root.content))
        except Exception as e:
            logger.error("Error writing preamble", file=str(preamble_file), error=str(e))
    process_children(root, output_folder)
    logger.info("Folder structure created", output=str(output_folder))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: {} input.md output_folder".format(sys.argv[0]))
        sys.exit(1)
    convert_md(Path(sys.argv[1]), Path(sys.argv[2]))
