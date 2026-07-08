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
        temperature=0.3,
    )


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def split_transcript(transcript: str) -> list[str]:
    return _split_text(transcript, TRANSCRIPT_CHUNK_SIZE, TRANSCRIPT_CHUNK_OVERLAP)


def _build_map_chain(llm):
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    return map_prompt | llm | StrOutputParser()


def _build_reduce_chain(llm):
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    return reduce_prompt | llm | StrOutputParser()


def _hierarchical_reduce(text_blocks: list[str], reduce_chain) -> str:
    current_blocks = [block for block in text_blocks if block.strip()]
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
            merged_blocks.append(reduce_chain.invoke({"text": block}))
        current_blocks = merged_blocks

    return current_blocks[0]


def summarize(transcript: str) -> str:
    llm = get_llm()
    map_chain = _build_map_chain(llm)
    reduce_chain = _build_reduce_chain(llm)

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    return _hierarchical_reduce(chunk_summaries, reduce_chain)


def generate_title(transcipt: str) -> str:
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()
    return title_chain.invoke({"text": transcipt[:2000]})
