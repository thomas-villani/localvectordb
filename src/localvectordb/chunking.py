# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/chunking.py
"""
Position-tracking chunking system for LocalVectorDB v1.0

This module provides chunkers that track exact positions in the original document,
enabling perfect reconstruction and precise highlighting.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type, Union

import tiktoken

from localvectordb.core import Chunk, ChunkPosition


class PositionTrackingChunker(ABC):
    """Base class for chunkers that track exact positions"""

    def __init__(self, max_tokens: int = 500, overlap: int = 0, **kwargs):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks with position tracking"""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def _calculate_line_column(self, text: str, position: int) -> Tuple[int, int]:
        """Calculate line and column for a character position"""
        if position > len(text):
            position = len(text)

        before = text[:position]
        lines = before.split('\n')
        line = len(lines)

        last_line = lines[-1]
        if last_line:
            column = len(last_line) + 1
        else:
            column = 1
        return line, column

    def _create_chunk(self, text: str, start: int, end: int, index: int) -> Chunk:
        """Create a chunk with position tracking"""
        content = text[start:end]
        line, column = self._calculate_line_column(text, start)
        end_line, end_column = self._calculate_line_column(text, end)

        position = ChunkPosition(
            start=start,
            end=end,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column
        )

        return Chunk(
            content=content,
            position=position,
            tokens=self.count_tokens(content),
            index=index
        )

    def _ensure_chunks_within_limit(self, chunks: List[Chunk], text: str) -> List[Chunk]:
        """Validate all chunks are within max_tokens limit, splitting oversized ones.

        This is a defensive safeguard to ensure no chunk exceeds max_tokens,
        regardless of the chunking strategy used. If any chunk exceeds the limit,
        it is split using character-level chunking.

        Args:
            chunks: List of chunks to validate
            text: Original text (for position recalculation if needed)

        Returns:
            List of chunks, all within max_tokens limit
        """
        result = []
        for chunk in chunks:
            if chunk.tokens <= self.max_tokens:
                result.append(chunk)
            else:
                # Split oversized chunk using CharChunker
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Chunk {chunk.index} exceeds max_tokens ({chunk.tokens} > {self.max_tokens}). "
                    f"Splitting with CharChunker."
                )

                # Create a CharChunker with the same max_tokens
                char_chunker = CharChunker(self.max_tokens, overlap=0)
                sub_chunks = char_chunker.chunk(chunk.content)

                # Adjust positions and indices for sub-chunks
                base_start = chunk.position.start
                base_index = chunk.index
                for i, sub_chunk in enumerate(sub_chunks):
                    # Recalculate positions relative to original text
                    new_start = base_start + sub_chunk.position.start
                    new_end = base_start + sub_chunk.position.end
                    line, column = self._calculate_line_column(text, new_start)
                    end_line, end_column = self._calculate_line_column(text, new_end)

                    sub_chunk.position = ChunkPosition(
                        start=new_start,
                        end=new_end,
                        line=line,
                        column=column,
                        end_line=end_line,
                        end_column=end_column
                    )
                    # Note: index will need to be renumbered by caller if needed
                    sub_chunk.index = base_index + i

                result.extend(sub_chunks)

        # Renumber indices to be sequential
        for i, chunk in enumerate(result):
            chunk.index = i

        return result


class SentenceChunker(PositionTrackingChunker):
    """Chunk by sentences while preserving boundaries"""
    sentence_pattern = re.compile(
        r'(?<=[.!?])\s+|(?<=[.!?]")(?=\s+[A-Z])|(?<=[.!?])\n+',
        re.MULTILINE
    )

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by sentences"""
        if not text.strip():
            return []

        # Find sentence boundaries
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [self._create_chunk(text, 0, len(text), 0)]

        chunks = []
        chunk_index = 0
        i = 0

        while i < len(sentences):
            # Start building a chunk
            chunk_sentences = []
            chunk_tokens = 0
            start_idx = i

            # Add sentences until we hit the token limit
            while i < len(sentences):
                sentence_start, sentence_end, sentence_text = sentences[i]
                sentence_tokens = self.count_tokens(sentence_text)

                # If this single sentence exceeds max_tokens, we need to split it
                if sentence_tokens > self.max_tokens and not chunk_sentences:
                    # Split the sentence by words
                    word_chunks = self._split_sentence_by_words(
                        text, sentence_start, sentence_end, chunk_index
                    )
                    chunks.extend(word_chunks)
                    chunk_index += len(word_chunks)
                    i += 1
                    break

                # If adding this sentence would exceed the limit
                if chunk_tokens + sentence_tokens > self.max_tokens and chunk_sentences:
                    break

                chunk_sentences.append((sentence_start, sentence_end, sentence_text))
                chunk_tokens += sentence_tokens
                i += 1

            # Create chunk if we have sentences
            if chunk_sentences:
                start_pos = chunk_sentences[0][0]
                end_pos = chunk_sentences[-1][1]
                chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

                # Apply overlap by setting next chunk start position
                if self.overlap > 0 and i < len(sentences):
                    sentences_processed = i - start_idx
                    overlap_count = min(self.overlap, sentences_processed - 1)
                    i = start_idx + max(1, sentences_processed - overlap_count)

        return self._ensure_chunks_within_limit(chunks, text)

    def _split_into_sentences(self, text: str) -> List[Tuple[int, int, str]]:
        """Split text into sentences with positions"""
        sentences = []
        last_end = 0

        for match in self.sentence_pattern.finditer(text):
            start = last_end
            end = match.start()

            if start < end:
                sentence_text = text[start:end].strip()
                if sentence_text:
                    sentences.append((start, end, sentence_text))

            last_end = match.end()

        # Add final sentence if any
        if last_end < len(text):
            final_text = text[last_end:].strip()
            if final_text:
                sentences.append((last_end, len(text), final_text))

        return sentences

    def _split_sentence_by_words(
            self,
            text: str,
            start: int,
            end: int,
            base_index: int
    ) -> List[Chunk]:
        """Split a long sentence by words while preserving whitespace"""
        # Extract the sentence text
        sentence_text = text[start:end]

        # Find word boundaries within this sentence
        word_pattern = re.compile(r'\S+')
        words = []

        # Find words and their absolute positions in the original text
        for match in word_pattern.finditer(sentence_text):
            word_start = start + match.start()  # Absolute position in original text
            word_end = start + match.end()  # Absolute position in original text
            word_text = match.group()
            words.append((word_start, word_end, word_text))

        if not words:
            # No words found, return the sentence as-is
            return [self._create_chunk(text, start, end, base_index)]

        chunks = []
        chunk_index = base_index
        i = 0

        while i < len(words):
            chunk_words = []

            # Add words until we hit the token limit
            while i < len(words):
                word_start, word_end, word_text = words[i]

                # For token counting, consider the text that would actually be in the chunk
                if chunk_words:
                    test_start = chunk_words[0][0]  # Start of first word in chunk
                else:
                    test_start = word_start

                # Determine where this chunk would end if we include this word
                if i + 1 < len(words):
                    # Next word exists, chunk would end just before next word
                    test_end = words[i + 1][0]
                else:
                    # This is the last word in sentence, chunk would end at end of sentence
                    test_end = end

                test_text = text[test_start:test_end]
                test_tokens = self.count_tokens(test_text)

                if test_tokens > self.max_tokens and chunk_words:
                    break

                chunk_words.append((word_start, word_end, word_text))
                i += 1

            # Create chunk if we have words
            if chunk_words:
                chunk_start = chunk_words[0][0]  # Start of first word

                # Determine end position: either start of next word or end of sentence
                if i < len(words):
                    chunk_end = words[i][0]  # Start of next word
                else:
                    chunk_end = end  # End of sentence

                chunk = self._create_chunk(text, chunk_start, chunk_end, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

        return chunks


class TokenChunker(PositionTrackingChunker):
    """Chunk by token boundaries with position tracking"""

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by token boundaries"""
        if not text.strip():
            return []

        # Encode the text to get tokens
        tokens = self.encoding.encode(text)

        if len(tokens) <= self.max_tokens:
            return [self._create_chunk(text, 0, len(text), 0)]

        chunks = []
        chunk_index = 0

        # Calculate stride (how far to advance for each chunk)
        stride = max(1, self.max_tokens - self.overlap)

        for i in range(0, len(tokens), stride):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + self.max_tokens]

            # Decode to get text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Find the actual position in the original text
            # This is approximate since token boundaries don't perfectly align with character boundaries
            start_pos = self._estimate_position(text, tokens, i)
            end_pos = min(start_pos + len(chunk_text), len(text))

            chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
            chunks.append(chunk)
            chunk_index += 1

            # If we've reached the end, break
            if end_pos >= len(text):
                break

        return self._ensure_chunks_within_limit(chunks, text)

    def _estimate_position(self, text: str, all_tokens: List[int], token_index: int) -> int:
        """Estimate character position from token index"""
        if token_index == 0:
            return 0

        # Decode tokens up to this point to estimate position
        prefix_tokens = all_tokens[:token_index]
        prefix_text = self.encoding.decode(prefix_tokens)

        # Find where this prefix ends in the original text
        return min(len(prefix_text), len(text))


class WordChunker(PositionTrackingChunker):
    """Chunk by word boundaries while preserving all whitespace"""

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by word boundaries while preserving whitespace"""
        if not text.strip():
            return []

        # Split into words while preserving positions - using \S+ to find word boundaries
        word_pattern = re.compile(r'\S+')
        words = [(m.start(), m.end(), m.group()) for m in word_pattern.finditer(text)]

        if not words:
            return [self._create_chunk(text, 0, len(text), 0)]

        chunks = []
        chunk_index = 0
        i = 0

        while i < len(words):
            chunk_words = []
            start_idx = i

            # Add words until we hit the token limit
            while i < len(words):
                word_start, word_end, word_text = words[i]

                # For token counting, we need to consider the text that would actually
                # be in the chunk, including trailing whitespace
                if chunk_words:
                    test_start = chunk_words[0][0]  # Start of first word in chunk
                else:
                    test_start = word_start

                # Determine where this chunk would end if we include this word
                if i + 1 < len(words):
                    # Next word exists, chunk would end just before next word
                    test_end = words[i + 1][0]
                else:
                    # This is the last word, chunk would end at end of text
                    test_end = len(text)

                test_text = text[test_start:test_end]
                test_tokens = self.count_tokens(test_text)

                if test_tokens > self.max_tokens and chunk_words:
                    break

                chunk_words.append((word_start, word_end, word_text))
                i += 1

            # Create chunk if we have words
            if chunk_words:
                start_pos = chunk_words[0][0]  # Start of first word

                # Determine end position: either start of next word or end of text
                if i < len(words):
                    end_pos = words[i][0]  # Start of next word
                else:
                    end_pos = len(text)  # End of text

                chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

                # Apply overlap by backing up some words
                if self.overlap > 0 and i < len(words):
                    words_processed = i - start_idx
                    overlap_count = min(self.overlap, words_processed - 1)
                    # Ensure we always make progress (at least 1 word forward)
                    i = start_idx + max(1, words_processed - overlap_count)

        return self._ensure_chunks_within_limit(chunks, text)


class LineChunker(PositionTrackingChunker):
    """Chunk by line boundaries"""

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by line boundaries"""
        if not text.strip():
            return []

        lines = text.splitlines(keepends=True)
        line_positions = []

        # Calculate line positions
        current_pos = 0
        for line in lines:
            start_pos = current_pos
            end_pos = current_pos + len(line)
            line_positions.append((start_pos, end_pos, line))
            current_pos = end_pos

        if not line_positions:
            return [self._create_chunk(text, 0, len(text), 0)]

        chunks = []
        chunk_index = 0
        i = 0

        while i < len(line_positions):
            chunk_lines = []
            chunk_tokens = 0
            start_idx = i

            # Add lines until we hit the token limit
            while i < len(line_positions):
                line_start, line_end, line_text = line_positions[i]
                line_tokens = self.count_tokens(line_text)

                # If single line exceeds max_tokens, split it by words
                if line_tokens > self.max_tokens and not chunk_lines:
                    word_chunks = self._split_line_by_words(
                        text, line_start, line_end, chunk_index
                    )
                    chunks.extend(word_chunks)
                    chunk_index += len(word_chunks)
                    i += 1
                    break

                if chunk_tokens + line_tokens > self.max_tokens and chunk_lines:
                    break

                chunk_lines.append((line_start, line_end, line_text))
                chunk_tokens += line_tokens
                i += 1

            # Create chunk if we have lines
            if chunk_lines:
                start_pos = chunk_lines[0][0]
                end_pos = chunk_lines[-1][1]
                chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

                # Apply overlap by setting next chunk start position
                # Ensure we always make progress (at least 1 line forward)
                if self.overlap > 0 and i < len(line_positions):
                    lines_processed = i - start_idx
                    overlap_count = min(self.overlap, lines_processed - 1)
                    i = start_idx + max(1, lines_processed - overlap_count)

        return self._ensure_chunks_within_limit(chunks, text)

    def _split_line_by_words(
            self,
            text: str,
            start: int,
            end: int,
            base_index: int
    ) -> List[Chunk]:
        """Split a long line by words"""
        line_text = text[start:end]
        word_chunker = WordChunker(self.max_tokens, self.overlap)
        word_chunks = word_chunker.chunk(line_text)

        # Adjust positions to be relative to the full text
        for chunk in word_chunks:
            chunk.position.start += start
            chunk.position.end += start
            chunk.index = base_index
            base_index += 1

        return word_chunks


class CharChunker(PositionTrackingChunker):
    """Chunk by character boundaries with exact position tracking"""

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by character boundaries"""
        if not text.strip():
            return []

        chunks = []
        chunk_index = 0
        start_pos = 0

        while start_pos < len(text):
            # Find the end position for this chunk
            end_pos = start_pos
            current_chunk = ""

            # Grow the chunk character by character until we hit max_tokens
            while end_pos < len(text):
                next_char = text[end_pos]
                test_chunk = current_chunk + next_char

                if self.count_tokens(test_chunk) > self.max_tokens and current_chunk:
                    break

                current_chunk = test_chunk
                end_pos += 1

            # Ensure we make progress even if first character exceeds max_tokens
            if end_pos == start_pos:
                end_pos = start_pos + 1

            # Create the chunk
            chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
            chunks.append(chunk)
            chunk_index += 1

            # Calculate next start position with overlap
            if end_pos >= len(text):
                break

            # For character overlap, move forward by (chunk_length - overlap)
            chunk_length = end_pos - start_pos
            advance = max(1, chunk_length - self.overlap)
            start_pos += advance

        return chunks


class ParagraphChunker(PositionTrackingChunker):
    """Chunk by paragraph boundaries"""
    paragraph_pattern = re.compile(r'\n\s*\n', re.MULTILINE)

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by paragraph boundaries"""
        if not text.strip():
            return []

        # Find paragraph boundaries
        paragraphs = self._split_into_paragraphs(text)

        if not paragraphs:
            return [self._create_chunk(text, 0, len(text), 0)]

        chunks = []
        chunk_index = 0
        i = 0

        while i < len(paragraphs):
            chunk_paragraphs = []
            chunk_tokens = 0
            start_idx = i

            # Add paragraphs until we hit the token limit
            while i < len(paragraphs):
                para_start, para_end, para_text = paragraphs[i]
                para_tokens = self.count_tokens(para_text)

                # If single paragraph exceeds max_tokens, split it by sentences
                if para_tokens > self.max_tokens and not chunk_paragraphs:
                    sentence_chunks = self._split_paragraph_by_sentences(
                        text, para_start, para_end, chunk_index
                    )
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    i += 1
                    break

                if chunk_tokens + para_tokens > self.max_tokens and chunk_paragraphs:
                    break

                chunk_paragraphs.append((para_start, para_end, para_text))
                chunk_tokens += para_tokens
                i += 1

            # Create chunk if we have paragraphs
            if chunk_paragraphs:
                start_pos = chunk_paragraphs[0][0]
                end_pos = chunk_paragraphs[-1][1]
                chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

                # Apply overlap by setting next chunk start position
                # Ensure we always make progress (at least 1 paragraph forward)
                if self.overlap > 0 and i < len(paragraphs):
                    paragraphs_processed = i - start_idx
                    overlap_count = min(self.overlap, paragraphs_processed - 1)
                    i = start_idx + max(1, paragraphs_processed - overlap_count)

        return self._ensure_chunks_within_limit(chunks, text)

    def _split_into_paragraphs(self, text: str) -> List[Tuple[int, int, str]]:
        """Split text into paragraphs with positions"""
        paragraphs = []
        last_end = 0

        for match in self.paragraph_pattern.finditer(text):
            start = last_end
            end = match.start()

            if start < end:
                para_text = text[start:end].strip()
                if para_text:
                    paragraphs.append((start, end, para_text))

            last_end = match.end()

        # Add final paragraph if any
        if last_end < len(text):
            final_text = text[last_end:].strip()
            if final_text:
                paragraphs.append((last_end, len(text), final_text))

        return paragraphs

    def _split_paragraph_by_sentences(
            self,
            text: str,
            start: int,
            end: int,
            base_index: int
    ) -> List[Chunk]:
        """Split a long paragraph by sentences"""
        para_text = text[start:end]
        sentence_chunker = SentenceChunker(self.max_tokens, self.overlap)
        sentence_chunks = sentence_chunker.chunk(para_text)

        # Adjust positions to be relative to the full text
        for chunk in sentence_chunks:
            chunk.position.start += start
            chunk.position.end += start
            chunk.index = base_index
            base_index += 1

        return sentence_chunks


class SectionChunker(PositionTrackingChunker):
    """Chunk by section headers (markdown-style)"""
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(self, max_tokens: int = 500, overlap: int = 0):
        if overlap != 0:
            raise ValueError("`overlap` must be 0 for SectionChunker")
        # Sections typically don't overlap as they're logical units
        # Force overlap to 0
        super().__init__(max_tokens, 0)

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by section headers"""
        if not text.strip():
            return []

        # Use a simple approach: treat headers as preferred break points
        # but don't let headers get orphaned

        # First, identify all header positions
        headers = list(self.header_pattern.finditer(text))
        header_positions = {h.start() for h in headers}

        # If no headers, just use paragraph chunking
        if not headers:
            if self.count_tokens(text) <= self.max_tokens:
                return [self._create_chunk(text, 0, len(text), 0)]
            else:
                para_chunker = ParagraphChunker(self.max_tokens, 0)
                return para_chunker.chunk(text)

        chunks = []
        chunk_index = 0
        current_pos = 0

        while current_pos < len(text):
            chunk_start = current_pos
            chunk_end = current_pos
            chunk_tokens = 0
            last_good_break = current_pos

            # Try to build a chunk up to max_tokens
            while chunk_end < len(text):
                # Find next potential break point (paragraph, line break, or end)
                next_break = chunk_end

                # Look for next line break
                next_newline = text.find('\n', chunk_end + 1)
                if next_newline == -1:
                    next_break = len(text)
                else:
                    next_break = next_newline + 1

                # Check tokens up to this break
                test_text = text[chunk_start:next_break]
                test_tokens = self.count_tokens(test_text)

                if test_tokens > self.max_tokens:
                    # We've exceeded the limit
                    if chunk_tokens == 0:
                        # Even the first segment is too large
                        # We need to break within it
                        if chunk_end in header_positions:
                            # We're at a header - find the end of the header line
                            header_end = text.find('\n', chunk_end)
                            if header_end == -1:
                                header_end = len(text)
                            else:
                                header_end += 1

                            # Include at least the header and some content
                            # Find a good break point after the header
                            search_start = header_end
                            while search_start < len(text):
                                search_end = text.find('\n', search_start)
                                if search_end == -1:
                                    search_end = len(text)
                                else:
                                    search_end += 1

                                if self.count_tokens(text[chunk_start:search_end]) <= self.max_tokens:
                                    chunk_end = search_end
                                    search_start = search_end
                                else:
                                    break

                            if chunk_end == chunk_start:
                                # Couldn't find any good break, just take what we can
                                chunk_end = header_end
                        else:
                            # Not at a header, break at word boundary
                            # This is a fallback for very large paragraphs
                            words = text[chunk_start:next_break].split()
                            accumulated = []
                            for word in words:
                                accumulated.append(word)
                                if self.count_tokens(' '.join(accumulated)) > self.max_tokens:
                                    accumulated.pop()
                                    break

                            if accumulated:
                                partial_text = ' '.join(accumulated)
                                chunk_end = chunk_start + len(partial_text)
                            else:
                                # Single word too large? Just take what we can
                                chunk_end = next_break
                    else:
                        # We have some content already
                        # Check if the next position is a header
                        if chunk_end in header_positions and last_good_break > chunk_start:
                            # Don't orphan the header - break before it
                            chunk_end = last_good_break
                        else:
                            # Use the last good break point
                            chunk_end = last_good_break
                    break

                # This break point fits
                chunk_tokens = test_tokens
                last_good_break = next_break
                chunk_end = next_break

                # If we're at a header position, mark it as a preferred break
                # unless it would orphan the header
                if chunk_end in header_positions:
                    # Look ahead to see if the section is small enough to include whole
                    next_header_pos = None
                    for h in headers:
                        if h.start() > chunk_end:
                            next_header_pos = h.start()
                            break

                    if next_header_pos is None:
                        next_header_pos = len(text)

                    # If the whole section fits, include it
                    section_text = text[chunk_end:next_header_pos]
                    if chunk_tokens + self.count_tokens(section_text) <= self.max_tokens:
                        chunk_end = next_header_pos
                        chunk_tokens = self.count_tokens(text[chunk_start:chunk_end])
                    # Otherwise, we'll break here before the header
                    else:
                        break

            # Create the chunk
            if chunk_end > chunk_start:
                chunk = self._create_chunk(text, chunk_start, chunk_end, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
                current_pos = chunk_end
            else:
                # Shouldn't happen, but safeguard against infinite loop
                current_pos = min(current_pos + 1, len(text))

        return self._ensure_chunks_within_limit(chunks, text)

    def _split_into_sections(self, text: str) -> List[Tuple[int, int, str, int]]:
        """Split text into sections based on headers"""
        sections = []
        headers = list(self.header_pattern.finditer(text))

        if not headers:
            # No headers found, treat entire text as one section
            return [(0, len(text), text, 0)]

        # Handle text before first header if it exists
        if headers[0].start() > 0:
            pre_text = text[:headers[0].start()].strip()
            if pre_text:
                sections.append((0, headers[0].start(), text[0:headers[0].start()], 0))

        # Process each header and its content
        for i, header_match in enumerate(headers):
            header_start = header_match.start()
            header_level = len(header_match.group(1))  # Number of # characters

            # Find where this section's content ends
            # Default to the start of the next header or end of text
            if i + 1 < len(headers):
                section_end = headers[i + 1].start()
            else:
                section_end = len(text)

            # Get the section text (including the header)
            section_text = text[header_start:section_end]

            sections.append((header_start, section_end, section_text, header_level))

        return sections

    def _split_large_section(
            self,
            text: str,
            start: int,
            end: int,
            base_index: int,
            header_level: int
    ) -> List[Chunk]:
        """Split a large section into smaller chunks"""
        # Not used in the new implementation, but kept for compatibility
        section_text = text[start:end]

        # Use paragraph chunker
        para_chunker = ParagraphChunker(self.max_tokens, 0)
        para_chunks = para_chunker.chunk(section_text)

        # Adjust positions to be relative to the full text
        for chunk in para_chunks:
            chunk.position.start += start
            chunk.position.end += start
            chunk.index = base_index
            base_index += 1

        return para_chunks


class CodeBlockChunker(PositionTrackingChunker):
    """Chunk code while preserving logical code blocks"""

    def __init__(self, max_tokens: int = 500, overlap: int = 0, language: Optional[str] = None, **kwargs):
        super().__init__(max_tokens, overlap, **kwargs)
        self.language = language

    def chunk(self, text: str) -> List[Chunk]:
        """Split code while preserving logical blocks"""
        if not text.strip():
            return []

        # Check if the entire text fits within the token limit
        if self.count_tokens(text) <= self.max_tokens:
            return [self._create_chunk(text, 0, len(text), 0)]

        # Detect language if not specified
        if not self.language:
            self.language = self._detect_language(text)

        # Split the text into lines for processing
        lines = text.splitlines()

        # Get language-specific block patterns
        block_start_patterns, block_end_patterns = self._get_language_patterns(self.language)

        # Find code blocks based on language patterns
        blocks = self._identify_code_blocks(lines, self.language, block_start_patterns, block_end_patterns)

        # Create chunks from code blocks
        chunks = self._create_chunks_from_blocks(text, lines, blocks)

        return self._ensure_chunks_within_limit(chunks, text)

    def _detect_language(self, text: str) -> str:
        """Detect programming language from code patterns"""
        # Take the first 1000 characters for detection (to avoid performance issues)
        sample = text[:1000]

        # Define pattern signatures for common languages
        signatures = {
            'python': {
                'patterns': [
                    r'^\s*def\s+\w+\s*\(.*\):',
                    r'^\s*class\s+\w+(\s*\(.*\))?:',
                    r'^\s*import\s+\w+',
                    r'^\s*from\s+[\w\.]+\s+import',
                    r'^\s*@\w+'
                ],
                'keywords': ['def', 'class', 'import', 'from', 'with', 'as', 'if', 'elif', 'else', 'for', 'while']
            },
            'javascript': {
                'patterns': [
                    r'function\s+\w+\s*\(.*\)\s*{',
                    r'const\s+\w+\s*=',
                    r'let\s+\w+\s*=',
                    r'var\s+\w+\s*=',
                    r'import\s+.*\s+from',
                    r'=>'
                ],
                'keywords': ['function', 'const', 'let', 'var', 'import', 'export', 'class', 'return']
            },
            'java': {
                'patterns': [
                    r'public\s+class',
                    r'private\s+\w+[\s\w]*\(.*\)\s*{',
                    r'protected\s+\w+[\s\w]*\(.*\)\s*{',
                    r'import\s+[\w\.]+;'
                ],
                'keywords': ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements']
            },
            'c': {
                'patterns': [
                    r'#include',
                    r'int\s+main\s*\(.*\)\s*{',
                    r'void\s+\w+\s*\(.*\)\s*{'
                ],
                'keywords': ['int', 'char', 'void', 'struct', 'typedef', 'return']
            },
            'cpp': {
                'patterns': [
                    r'#include',
                    r'namespace\s+\w+\s*{',
                    r'class\s+\w+\s*{',
                    r'std::'
                ],
                'keywords': ['class', 'namespace', 'template', 'typename', 'auto', 'public', 'private']
            },
            'go': {
                'patterns': [
                    r'package\s+\w+',
                    r'func\s+\w+\s*\(.*\)\s*{',
                    r'import\s+\('
                ],
                'keywords': ['func', 'package', 'import', 'type', 'struct', 'interface']
            },
            'ruby': {
                'patterns': [
                    r'def\s+\w+',
                    r'class\s+\w+',
                    r'module\s+\w+',
                    r'require\s+',
                    r'end$'
                ],
                'keywords': ['def', 'class', 'module', 'require', 'end', 'attr_']
            },
            'php': {
                'patterns': [
                    r'<\?php',
                    r'function\s+\w+\s*\(.*\)\s*{',
                    r'\$\w+\s*='
                ],
                'keywords': ['function', 'class', 'echo', 'public', 'private', 'namespace']
            }
        }

        # Count matches for each language
        scores = {lang: 0 for lang in signatures}

        # Check for language patterns
        for lang, sig in signatures.items():
            for pattern in sig['patterns']:
                matches = re.findall(pattern, sample, re.MULTILINE)
                scores[lang] += len(matches) * 2  # Weight patterns more heavily

            # Check for keywords
            for keyword in sig.get('keywords', []):
                matches = re.findall(r'\b' + keyword + r'\b', sample)
                scores[lang] += len(matches)

        # Check for language-specific structural features
        if '    ' in sample and '{' not in sample[:100]:
            scores['python'] += 5  # Python-style indentation is distinctive

        if '{' in sample and '}' in sample:
            for lang in ['javascript', 'java', 'c', 'cpp', 'go', 'php']:
                scores[lang] += 2

        # Find language with highest score
        best_lang = max(scores.items(), key=lambda x: x[1])

        # If the best score is very low, default to generic
        if best_lang[1] < 3:
            return "generic"

        return best_lang[0]

    def _get_language_patterns(self, language: str) -> Tuple[List[str], List[str]]:
        """Get language-specific block start and end patterns"""
        patterns = {
            'python': {
                'starts': [
                    r'^\s*def\s+\w+\s*\(.*\):',
                    r'^\s*class\s+\w+(\s*\(.*\))?:',
                    r'^\s*if\s+.*:',
                    r'^\s*while\s+.*:',
                    r'^\s*for\s+.*:',
                    r'^\s*try:',
                    r'^\s*except.*:',
                    r'^\s*with.*:'
                ],
                'ends': []  # Python uses indentation
            },
            'javascript': {
                'starts': [
                    r'function\s+\w+\s*\(.*\)\s*{',
                    r'class\s+\w+(\s+extends\s+\w+)?\s*{',
                    r'[\w\.]+\s*=\s*function\s*\(.*\)\s*{',
                    r'[\w\.]+\s*=\s*\(.*\)\s*=>\s*{',
                    r'if\s*\(.*\)\s*{',
                    r'for\s*\(.*\)\s*{',
                    r'while\s*\(.*\)\s*{'
                ],
                'ends': [r'^(\s*)\}']
            },
            'java': {
                'starts': [
                    r'(public|private|protected)?\s*\w+(\s+\w+)?\s*\(.*\)\s*{',
                    r'class\s+\w+(\s+extends\s+\w+)?(\s+implements\s+[\w,\s]+)?\s*{',
                    r'interface\s+\w+\s*{',
                    r'if\s*\(.*\)\s*{',
                    r'for\s*\(.*\)\s*{',
                    r'while\s*\(.*\)\s*{'
                ],
                'ends': [r'^(\s*)\}']
            },
            'generic': {
                'starts': [
                    r'^\s*\w+\s*\(.*\)\s*{',
                    r'^\s*class\s+\w+\s*{',
                    r'^\s*if\s*\(.*\)\s*{',
                    r'^\s*for\s*\(.*\)\s*{',
                    r'^\s*while\s*\(.*\)\s*{'
                ],
                'ends': [r'^(\s*)\}', r'^(\s*)end']
            }
        }

        # Get patterns for the specified language or fall back to generic
        lang_patterns = patterns.get(language, patterns['generic'])
        return lang_patterns['starts'], lang_patterns['ends']

    @staticmethod
    def _identify_code_blocks(
            lines: List[str], language: str,
            start_patterns: List[str], end_patterns: List[str]
    ) -> List[dict]:
        """Identify code blocks based on language-specific patterns"""
        blocks = []
        i = 0

        # Special handling for Python (indentation-based)
        if language == 'python':
            while i < len(lines):
                # Look for block start
                block_start = None
                for pattern in start_patterns:
                    if re.match(pattern, lines[i]):
                        block_start = i
                        break

                if block_start is not None:
                    # Find the indentation level of the block start
                    indent_match = re.match(r'^(\s*)', lines[i])
                    start_indent = indent_match.group(1) if indent_match else ''

                    # Find where the block ends (when indentation returns to the same level)
                    j = i + 1
                    while j < len(lines):
                        # Skip blank lines or lines with only comments
                        if not lines[j].strip() or lines[j].strip().startswith('#'):
                            j += 1
                            continue

                        # Check if we're back to the original indentation level
                        indent_match = re.match(r'^(\s*)', lines[j])
                        current_indent = indent_match.group(1) if indent_match else ''

                        if len(current_indent) <= len(start_indent) and j > i + 1:
                            # This line has the same or less indentation, marking the end of the block
                            blocks.append({'start': block_start, 'end': j - 1, 'type': 'block'})
                            i = j - 1  # Will be incremented in the outer loop
                            break

                        j += 1

                    # If we reached the end of the file, add the final block
                    if j == len(lines):
                        blocks.append({'start': block_start, 'end': j - 1, 'type': 'block'})
                        i = j - 1

                i += 1

        # For bracket-based languages
        else:
            bracket_stack = []

            while i < len(lines):
                line = lines[i]

                # Count opening and closing brackets
                opening_brackets = line.count('{')
                closing_brackets = line.count('}')

                # Check for block starts
                is_block_start = False
                for pattern in start_patterns:
                    if re.search(pattern, line):
                        is_block_start = True

                        # Create a new block
                        if not bracket_stack:
                            blocks.append({
                                'start': i,
                                'end': None,
                                'bracket_depth': 1,
                                'type': 'block'
                            })
                            bracket_stack.append(len(blocks) - 1)  # Store the index of this block

                        break

                # Update bracket depth
                if opening_brackets > 0 and not is_block_start:
                    if not bracket_stack:
                        # This is a new block starting with just a {
                        blocks.append({
                            'start': i,
                            'end': None,
                            'bracket_depth': opening_brackets,
                            'type': 'block'
                        })
                        bracket_stack.append(len(blocks) - 1)
                    else:
                        # Update the bracket depth of the current block
                        block_idx = bracket_stack[-1]
                        if block_idx < len(blocks):
                            blocks[block_idx]['bracket_depth'] += opening_brackets

                # Check for block ends
                if closing_brackets > 0:
                    if bracket_stack:
                        block_idx = bracket_stack[-1]
                        if block_idx < len(blocks):
                            blocks[block_idx]['bracket_depth'] -= closing_brackets

                            # If this block is complete, mark its end
                            if blocks[block_idx]['bracket_depth'] <= 0:
                                blocks[block_idx]['end'] = i
                                bracket_stack.pop()

                i += 1

            # Close any unclosed blocks at EOF
            for block_idx in bracket_stack:
                if block_idx < len(blocks) and blocks[block_idx]['end'] is None:
                    blocks[block_idx]['end'] = len(lines) - 1

        # Collect all lines that are not part of a specific block
        covered_lines = set()
        for block in blocks:
            for j in range(block['start'], block['end'] + 1):
                covered_lines.add(j)

        # Add remaining lines as smaller chunks
        i = 0
        while i < len(lines):
            if i not in covered_lines:
                # Find the next uncovered range
                start = i
                while i < len(lines) and i not in covered_lines:
                    i += 1

                # Add as a non-block chunk if it contains code
                if any(line.strip() for line in lines[start:i]):
                    blocks.append({
                        'start': start,
                        'end': i - 1,
                        'type': 'non-block'
                    })
                continue
            i += 1

        # Sort blocks by start position
        blocks.sort(key=lambda x: x['start'])

        return blocks

    def _create_chunks_from_blocks(self, text: str, lines: List[str], blocks: List[dict]) -> List[Chunk]:
        """Create text chunks from code blocks respecting max_tokens"""
        if not blocks:
            return []

        chunks = []
        chunk_index = 0
        current_chunk_lines = []
        current_tokens = 0

        # Calculate line positions in the original text
        line_positions = []
        current_pos = 0
        for line in lines:
            line_positions.append(current_pos)
            current_pos += len(line) + 1  # +1 for newline character

        for block in blocks:
            # Get the block lines
            block_lines = lines[block['start']:block['end'] + 1]
            block_text = '\n'.join(block_lines)
            block_tokens = self.count_tokens(block_text)

            # If this block alone exceeds max_tokens, split it further
            if block_tokens > self.max_tokens:
                # First, finalize the current chunk
                if current_chunk_lines:
                    chunk_start_pos = line_positions[current_chunk_lines[0]['line_idx']]
                    chunk_end_pos = line_positions[current_chunk_lines[-1]['line_idx']] + len(
                        current_chunk_lines[-1]['line'])
                    chunk = self._create_chunk(text, chunk_start_pos, chunk_end_pos, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_lines = []
                    current_tokens = 0

                # Now split the large block using line-based chunking
                block_start_pos = line_positions[block['start']]
                block_end_pos = line_positions[block['end']] + len(lines[block['end']])

                # Use LineChunker for large blocks
                large_block_text = text[block_start_pos:block_end_pos]
                line_chunker = LineChunker(self.max_tokens, self.overlap)
                line_chunks = line_chunker.chunk(large_block_text)

                # Convert line chunks to proper chunks with adjusted positions
                for line_chunk in line_chunks:
                    chunk_start = block_start_pos + large_block_text.index(line_chunk.content)
                    chunk_end = chunk_start + len(line_chunk.content)
                    chunk = self._create_chunk(text, chunk_start, chunk_end, chunk_index)
                    chunks.append(chunk)
                    chunk_index += 1
                continue

            # Check if adding this block would exceed max_tokens
            if current_tokens + block_tokens > self.max_tokens and current_chunk_lines:
                # Finalize current chunk
                chunk_start_pos = line_positions[current_chunk_lines[0]['line_idx']]
                chunk_end_pos = line_positions[current_chunk_lines[-1]['line_idx']] + len(
                    current_chunk_lines[-1]['line'])
                chunk = self._create_chunk(text, chunk_start_pos, chunk_end_pos, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

                # Start a new chunk, with overlap if applicable
                if self.overlap > 0 and current_chunk_lines:
                    # Calculate how many lines to include for overlap
                    overlap_lines = min(self.overlap, len(current_chunk_lines))
                    current_chunk_lines = current_chunk_lines[-overlap_lines:]
                    current_tokens = self.count_tokens('\n'.join([line['line'] for line in current_chunk_lines]))
                else:
                    current_chunk_lines = []
                    current_tokens = 0

            # Add block to current chunk
            for i, line in enumerate(block_lines):
                current_chunk_lines.append({
                    'line': line,
                    'line_idx': block['start'] + i
                })
            current_tokens = self.count_tokens('\n'.join([line['line'] for line in current_chunk_lines]))

        # Add the final chunk if not empty
        if current_chunk_lines:
            chunk_start_pos = line_positions[current_chunk_lines[0]['line_idx']]
            chunk_end_pos = line_positions[current_chunk_lines[-1]['line_idx']] + len(current_chunk_lines[-1]['line'])
            chunk = self._create_chunk(text, chunk_start_pos, chunk_end_pos, chunk_index)
            chunks.append(chunk)

        return [chunk for chunk in chunks if chunk.content.strip()]


class ChunkerFactory:
    """Factory for creating chunkers"""

    CHUNKERS: dict[str, Type[PositionTrackingChunker]] = {
        'sentences': SentenceChunker,
        'tokens': TokenChunker,
        'words': WordChunker,
        'lines': LineChunker,
        'characters': CharChunker,
        'paragraphs': ParagraphChunker,
        'sections': SectionChunker,
        'code-blocks': CodeBlockChunker,
    }

    @classmethod
    def create_chunker(
            cls,
            method: Union[str, Type[PositionTrackingChunker]],
            max_tokens: int = 500,
            overlap: int = 0,
            **kwargs
    ) -> PositionTrackingChunker:
        """Create a chunker instance"""
        if isinstance(method, type) and hasattr(method, "chunk"):
            return method(max_tokens, overlap=overlap)
        elif not isinstance(method, str):
            raise TypeError("Error creating chunker: `method` must be `str` or `PositionTrackingChunker`, "
                            f"found: {type(method)}")
        if method not in cls.CHUNKERS:
            available = ', '.join(cls.CHUNKERS.keys())
            raise ValueError(f"Unknown chunking method: {method}. Available: {available}")

        if not isinstance(max_tokens, int):
            raise TypeError(f"`max_tokens` must be a positive integer, found: {type(max_tokens)}")
        if max_tokens <= 0:
            raise ValueError(f"`max_tokens` must be a positive integer, found: {max_tokens}")
        chunker_class = cls.CHUNKERS[method]

        # Map overlap parameter names based on chunker type
        if method == 'sentences':
            return chunker_class(max_tokens, overlap=overlap, **kwargs)
        elif method == 'words':
            return chunker_class(max_tokens, overlap=overlap, **kwargs)
        elif method == 'lines':
            return chunker_class(max_tokens, overlap=overlap, **kwargs)
        elif method == 'characters':
            return chunker_class(max_tokens, overlap=overlap, **kwargs)
        elif method == 'paragraphs':
            return chunker_class(max_tokens, overlap=overlap, **kwargs)
        elif method == 'code-blocks':
            return chunker_class(max_tokens, overlap=overlap, **kwargs)
        elif method == 'sections':
            # Sections typically don't overlap
            return chunker_class(max_tokens, overlap=0, **kwargs)
        else:
            return chunker_class(max_tokens, overlap, **kwargs)

    @classmethod
    def list_methods(cls) -> List[str]:
        """List available chunking methods"""
        return list(cls.CHUNKERS.keys())


def reconstruct_document(chunks: List[Chunk], original_length: int) -> str:
    """Perfectly reconstruct a document from its chunks"""
    if not chunks:
        return ""

    # Sort chunks by position
    sorted_chunks = sorted(chunks, key=lambda c: c.position.start)

    # Create a character array to fill
    result = [""] * original_length

    # Fill in the chunks
    for chunk in sorted_chunks:
        start = chunk.position.start
        end = chunk.position.end

        # Ensure we don't go out of bounds
        end = min(end, original_length)

        if start < original_length:
            result[start:end] = list(chunk.content[:end - start])

    # Join and return
    return "".join(result)
