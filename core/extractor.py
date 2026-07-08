# Action items, decisions, questions

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

TRANSCRIPT_CHUNK_SIZE = 3000
TRANSCRIPT_CHUNK_OVERLAP = 200
REDUCE_CHUNK_SIZE = 6000
REDUCE_CHUNK_OVERLAP = 400


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def _map_reduce_extract(transcript: str, chunk_prompt: str, combine_prompt: str) -> str:
    llm = get_llm()
    chunk_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", chunk_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    combine_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", combine_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    chunk_results = [
        chunk_chain.invoke({"text": chunk})
        for chunk in _split_text(transcript, TRANSCRIPT_CHUNK_SIZE, TRANSCRIPT_CHUNK_OVERLAP)
    ]

    current_blocks = [block for block in chunk_results if block.strip()]
    if not current_blocks:
        return ""

    while len(current_blocks) > 1:
        merged_blocks = []
        grouped_blocks = _split_text(
            "\n\n".join(current_blocks),
            REDUCE_CHUNK_SIZE,
            REDUCE_CHUNK_OVERLAP,
        )
        for block in grouped_blocks:
            merged_blocks.append(combine_chain.invoke({"text": block}))
        current_blocks = merged_blocks

    return current_blocks[0]


def extract_action_items(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract action items. For each provide task description, owner, and deadline "
        "(or 'Not specified' if missing). Format as a numbered list. If none are found, "
        "say 'No action items found.'",
        "You are an expert meeting analyst. Merge these partial action-item lists into one "
        "deduplicated numbered list. Preserve owners and deadlines when present. "
        "If nothing actionable exists, return 'No action items found.'.",
    )


def extract_key_decisions(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract all key decisions made. Format as a numbered list. If none are found, "
        "say 'No key decisions found.'",
        "You are an expert meeting analyst. Merge these partial decision lists into one "
        "deduplicated numbered list. If no decisions were actually made, return "
        "'No key decisions found.'.",
    )


def extract_questions(transcript: str) -> str:
    return _map_reduce_extract(
        transcript,
        "You are an expert meeting analyst. From this portion of a meeting transcript, "
        "extract unresolved questions or topics needing follow-up. Format as a numbered "
        "list. If none are found, say 'No open questions found.'",
        "You are an expert meeting analyst. Merge these partial follow-up question lists "
        "into one deduplicated numbered list. If nothing remains unresolved, return "
        "'No open questions found.'.",
    )
