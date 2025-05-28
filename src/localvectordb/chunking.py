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
import tiktoken
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from localvectordb.core import Chunk, ChunkPosition


class PositionTrackingChunker(ABC):
    """Base class for chunkers that track exact positions"""

    def __init__(self, max_tokens: int = 500, overlap: int = 0):
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

        lines = text[:position].split('\n')
        line = len(lines)
        column = len(lines[-1]) + 1 if lines else 1
        return line, column

    def _create_chunk(self, text: str, start: int, end: int, index: int) -> Chunk:
        """Create a chunk with position tracking"""
        content = text[start:end]
        line, column = self._calculate_line_column(text, start)

        position = ChunkPosition(
            start=start,
            end=end,
            line=line,
            column=column
        )

        return Chunk(
            content=content,
            position=position,
            tokens=self.count_tokens(content),
            index=index
        )


class SentenceChunker(PositionTrackingChunker):
    """Chunk by sentences while preserving boundaries"""

    def __init__(self, max_tokens: int = 500, overlap_sentences: int = 1):
        super().__init__(max_tokens, overlap_sentences)
        self.sentence_pattern = re.compile(
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

                # Apply overlap by backing up
                if self.overlap > 0 and i < len(sentences):
                    overlap_count = min(self.overlap, len(chunk_sentences))
                    i -= overlap_count

        return chunks

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
        """Split a long sentence by words"""
        sentence = text[start:end]
        words = sentence.split()

        chunks = []
        chunk_index = base_index
        i = 0

        while i < len(words):
            chunk_words = []
            chunk_tokens = 0
            word_positions = []

            # Find positions of words in the sentence
            current_pos = start
            for word_idx, word in enumerate(words[i:], i):
                word_start = text.find(word, current_pos)
                if word_start == -1:
                    word_start = current_pos
                word_end = word_start + len(word)
                word_positions.append((word_start, word_end))
                current_pos = word_end

                word_tokens = self.count_tokens(word)

                if chunk_tokens + word_tokens > self.max_tokens and chunk_words:
                    break

                chunk_words.append(word)
                chunk_tokens += word_tokens
                i += 1

            if chunk_words:
                chunk_start = word_positions[0][0]
                chunk_end = word_positions[-1][1]
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

            # Adjust to actual character boundaries
            actual_chunk_text = text[start_pos:end_pos]

            chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
            chunks.append(chunk)
            chunk_index += 1

            # If we've reached the end, break
            if end_pos >= len(text):
                break

        return chunks

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
    """Chunk by word boundaries"""

    def __init__(self, max_tokens: int = 500, overlap_words: int = 10):
        super().__init__(max_tokens, overlap_words)

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by word boundaries"""
        if not text.strip():
            return []

        # Split into words while preserving positions
        word_pattern = re.compile(r'\S+')
        words = [(m.start(), m.end(), m.group()) for m in word_pattern.finditer(text)]

        if not words:
            return [self._create_chunk(text, 0, len(text), 0)]

        chunks = []
        chunk_index = 0
        i = 0

        while i < len(words):
            chunk_words = []
            chunk_tokens = 0

            # Add words until we hit the token limit
            while i < len(words):
                word_start, word_end, word_text = words[i]

                # Estimate tokens (include surrounding context)
                if chunk_words:
                    # Get text from first word start to current word end
                    test_start = chunk_words[0][0]
                    test_text = text[test_start:word_end]
                else:
                    test_text = word_text

                test_tokens = self.count_tokens(test_text)

                if test_tokens > self.max_tokens and chunk_words:
                    break

                chunk_words.append((word_start, word_end, word_text))
                chunk_tokens = test_tokens
                i += 1

            # Create chunk if we have words
            if chunk_words:
                start_pos = chunk_words[0][0]
                end_pos = chunk_words[-1][1]
                chunk = self._create_chunk(text, start_pos, end_pos, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

                # Apply overlap by backing up
                if self.overlap > 0 and i < len(words):
                    overlap_count = min(self.overlap, len(chunk_words))
                    i -= overlap_count

        return chunks


class LineChunker(PositionTrackingChunker):
    """Chunk by line boundaries"""

    def __init__(self, max_tokens: int = 500, overlap_lines: int = 2):
        super().__init__(max_tokens, overlap_lines)

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

                # Apply overlap by backing up
                if self.overlap > 0 and i < len(line_positions):
                    overlap_count = min(self.overlap, len(chunk_lines))
                    i -= overlap_count

        return chunks

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


# Additional chunkers for the chunking.py module

class CharChunker(PositionTrackingChunker):
    """Chunk by character boundaries with exact position tracking"""

    def __init__(self, max_tokens: int = 500, overlap_chars: int = 50):
        super().__init__(max_tokens, overlap_chars)

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

    def __init__(self, max_tokens: int = 500, overlap_paragraphs: int = 1):
        super().__init__(max_tokens, overlap_paragraphs)
        # Match paragraph separators (double newlines or more)
        self.paragraph_pattern = re.compile(r'\n\s*\n', re.MULTILINE)

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

                # Apply overlap by backing up
                if self.overlap > 0 and i < len(paragraphs):
                    overlap_count = min(self.overlap, len(chunk_paragraphs))
                    i -= overlap_count

        return chunks

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

    def __init__(self, max_tokens: int = 500, overlap: int = 0):
        # Sections typically don't overlap as they're logical units
        super().__init__(max_tokens, overlap)
        # Match markdown headers (# ## ### etc.) at start of line
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def chunk(self, text: str) -> List[Chunk]:
        """Split text by section headers"""
        if not text.strip():
            return []

        # Find section boundaries
        sections = self._split_into_sections(text)

        if not sections:
            return [self._create_chunk(text, 0, len(text), 0)]

        chunks = []
        chunk_index = 0

        for section_start, section_end, section_text, header_level in sections:
            section_tokens = self.count_tokens(section_text)

            # If section exceeds max_tokens, split it further
            if section_tokens > self.max_tokens:
                # Try splitting by subsections first, then by paragraphs
                sub_chunks = self._split_large_section(
                    text, section_start, section_end, chunk_index, header_level
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                # Section fits within token limit
                chunk = self._create_chunk(text, section_start, section_end, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _split_into_sections(self, text: str) -> List[Tuple[int, int, str, int]]:
        """Split text into sections based on headers"""
        sections = []
        headers = list(self.header_pattern.finditer(text))

        if not headers:
            # No headers found, treat entire text as one section
            return [(0, len(text), text, 1)]

        # Process each header
        for i, header_match in enumerate(headers):
            header_start = header_match.start()
            header_level = len(header_match.group(1))  # Number of # characters

            # Find the end of this section
            if i + 1 < len(headers):
                next_header = headers[i + 1]
                next_level = len(next_header.group(1))

                # Section ends at next header of same or higher level
                section_end = next_header.start()
                for j in range(i + 1, len(headers)):
                    check_header = headers[j]
                    check_level = len(check_header.group(1))
                    if check_level <= header_level:
                        section_end = check_header.start()
                        break
                    if j == len(headers) - 1:
                        section_end = len(text)
            else:
                section_end = len(text)

            section_text = text[header_start:section_end].strip()
            if section_text:
                sections.append((header_start, section_end, section_text, header_level))

        # Handle text before first header
        if headers and headers[0].start() > 0:
            pre_text = text[:headers[0].start()].strip()
            if pre_text:
                sections.insert(0, (0, headers[0].start(), pre_text, 0))

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
        section_text = text[start:end]

        # Try splitting by paragraphs first
        para_chunker = ParagraphChunker(self.max_tokens, 0)  # No overlap for sections
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

    def __init__(self, max_tokens: int = 500, overlap_lines: int = 2, language: Optional[str] = None):
        super().__init__(max_tokens, overlap_lines)
        self.language = language or self._detect_language_hint()

        # Language-specific patterns for code blocks
        self.block_patterns = {
            'python': [
                r'^\s*(def\s+\w+.*?:)',
                r'^\s*(class\s+\w+.*?:)',
                r'^\s*(if\s+.*?:)',
                r'^\s*(for\s+.*?:)',
                r'^\s*(while\s+.*?:)',
                r'^\s*(with\s+.*?:)',
                r'^\s*(try\s*:)',
                r'^\s*(except.*?:)',
                r'^\s*(finally\s*:)',
            ],
            'javascript': [
                r'function\s+\w+\s*\([^)]*\)\s*{',
                r'class\s+\w+\s*{',
                r'const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*{',
                r'if\s*\([^)]*\)\s*{',
                r'for\s*\([^)]*\)\s*{',
                r'while\s*\([^)]*\)\s*{',
            ],
            'java': [
                r'(public|private|protected)?\s*(static\s+)?[^{]*\{',
                r'class\s+\w+[^{]*\{',
                r'interface\s+\w+[^{]*\{',
                r'if\s*\([^)]*\)\s*\{',
                r'for\s*\([^)]*\)\s*\{',
                r'while\s*\([^)]*\)\s*\{',
            ],
            'generic': [
                r'^\s*\w+.*?\{',
                r'^\s*(if|for|while).*?\{',
                r'^\s*(function|class|def)\s+\w+',
            ]
        }

    def chunk(self, text: str) -> List[Chunk]:
        """Split code while preserving logical blocks"""
        if not text.strip():
            return []

        # Detect language if not specified
        if not self.language:
            self.language = self._detect_language(text)

        # Find code blocks
        blocks = self._identify_code_blocks(text)

        if not blocks:
            # Fall back to line-based chunking
            line_chunker = LineChunker(self.max_tokens, self.overlap)
            return line_chunker.chunk(text)

        chunks = []
        chunk_index = 0

        for block_start, block_end, block_text, block_type in blocks:
            block_tokens = self.count_tokens(block_text)

            # If block exceeds max_tokens, split it further
            if block_tokens > self.max_tokens:
                sub_chunks = self._split_large_block(
                    text, block_start, block_end, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                # Block fits within token limit
                chunk = self._create_chunk(text, block_start, block_end, chunk_index)
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _detect_language_hint(self) -> str:
        """Get a language hint from context if available"""
        # This could be enhanced to detect from file extensions, etc.
        return 'generic'

    def _detect_language(self, text: str) -> str:
        """Detect programming language from code patterns"""
        sample = text[:1000]  # Sample first 1000 chars for detection

        scores = {
            'python': 0,
            'javascript': 0,
            'java': 0,
        }

        # Check for language-specific keywords and patterns
        if 'def ' in sample or 'import ' in sample or 'from ' in sample:
            scores['python'] += 3
        if 'function ' in sample or 'const ' in sample or 'let ' in sample:
            scores['javascript'] += 3
        if 'public class' in sample or 'private ' in sample or 'import java' in sample:
            scores['java'] += 3

        # Check indentation style
        if '    ' in sample and '{' not in sample[:200]:
            scores['python'] += 2
        if sample.count('{') > 2:
            scores['javascript'] += 1
            scores['java'] += 1

        # Return language with highest score, or generic
        best_lang = max(scores.items(), key=lambda x: x[1])
        return best_lang[0] if best_lang[1] > 2 else 'generic'

    def _identify_code_blocks(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Identify code blocks in the text"""
        blocks = []
        lines = text.splitlines(keepends=True)

        patterns = self.block_patterns.get(self.language, self.block_patterns['generic'])

        if self.language == 'python':
            # Python uses indentation for blocks
            blocks = self._identify_python_blocks(text, lines)
        else:
            # Brace-based languages
            blocks = self._identify_brace_blocks(text, lines, patterns)

        # Fill gaps with non-block content
        blocks = self._fill_gaps(text, blocks)

        return sorted(blocks, key=lambda x: x[0])

    def _identify_python_blocks(self, text: str, lines: List[str]) -> List[Tuple[int, int, str, str]]:
        """Identify Python code blocks based on indentation"""
        blocks = []
        current_pos = 0

        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)

            # Check if this line starts a block
            is_block_start = False
            for pattern in self.block_patterns['python']:
                if re.match(pattern, line, re.MULTILINE):
                    is_block_start = True
                    break

            if is_block_start:
                # Find the indentation level
                indent_match = re.match(r'^(\s*)', line)
                start_indent = len(indent_match.group(1)) if indent_match else 0

                # Find where this block ends
                block_end = line_end
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    next_pos = sum(len(lines[k]) for k in range(j))

                    # Skip empty lines and comments
                    if not next_line.strip() or next_line.strip().startswith('#'):
                        block_end = next_pos + len(next_line)
                        continue

                    # Check indentation
                    next_indent_match = re.match(r'^(\s*)', next_line)
                    next_indent = len(next_indent_match.group(1)) if next_indent_match else 0

                    if next_indent <= start_indent:
                        break

                    block_end = next_pos + len(next_line)

                block_text = text[line_start:block_end]
                blocks.append((line_start, block_end, block_text, 'function'))

            current_pos = line_end

        return blocks

    def _identify_brace_blocks(
            self,
            text: str,
            lines: List[str],
            patterns: List[str]
    ) -> List[Tuple[int, int, str, str]]:
        """Identify code blocks based on braces"""
        blocks = []
        current_pos = 0

        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)

            # Check if this line starts a block
            is_block_start = False
            for pattern in patterns:
                if re.search(pattern, line, re.MULTILINE):
                    is_block_start = True
                    break

            if is_block_start and '{' in line:
                # Find matching closing brace
                brace_count = line.count('{') - line.count('}')
                block_end = line_end

                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    next_pos = sum(len(lines[k]) for k in range(j))

                    brace_count += next_line.count('{') - next_line.count('}')
                    block_end = next_pos + len(next_line)

                    if brace_count <= 0:
                        break

                block_text = text[line_start:block_end]
                blocks.append((line_start, block_end, block_text, 'function'))

            current_pos = line_end

        return blocks

    def _fill_gaps(self, text: str, blocks: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        """Fill gaps between blocks with non-block content"""
        if not blocks:
            return [(0, len(text), text, 'code')]

        filled_blocks = []
        last_end = 0

        for start, end, content, block_type in blocks:
            # Add gap before this block
            if start > last_end:
                gap_text = text[last_end:start].strip()
                if gap_text:
                    filled_blocks.append((last_end, start, gap_text, 'code'))

            filled_blocks.append((start, end, content, block_type))
            last_end = end

        # Add final gap
        if last_end < len(text):
            gap_text = text[last_end:].strip()
            if gap_text:
                filled_blocks.append((last_end, len(text), gap_text, 'code'))

        return filled_blocks

    def _split_large_block(
            self,
            text: str,
            start: int,
            end: int,
            base_index: int
    ) -> List[Chunk]:
        """Split a large code block into smaller chunks"""
        block_text = text[start:end]

        # Fall back to line-based chunking for large blocks
        line_chunker = LineChunker(self.max_tokens, self.overlap)
        line_chunks = line_chunker.chunk(block_text)

        # Adjust positions to be relative to the full text
        for chunk in line_chunks:
            chunk.position.start += start
            chunk.position.end += start
            chunk.index = base_index
            base_index += 1

        return line_chunks


class ChunkerFactory:
    """Factory for creating chunkers"""

    CHUNKERS = {
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
            method: str,
            max_tokens: int = 500,
            overlap: int = 0,
            **kwargs
    ) -> PositionTrackingChunker:
        """Create a chunker instance"""
        if method not in cls.CHUNKERS:
            available = ', '.join(cls.CHUNKERS.keys())
            raise ValueError(f"Unknown chunking method: {method}. Available: {available}")

        chunker_class = cls.CHUNKERS[method]

        # Map overlap parameter names based on chunker type
        if method == 'sentences':
            return chunker_class(max_tokens, overlap_sentences=overlap, **kwargs)
        elif method == 'words':
            return chunker_class(max_tokens, overlap_words=overlap, **kwargs)
        elif method == 'lines':
            return chunker_class(max_tokens, overlap_lines=overlap, **kwargs)
        elif method == 'characters':
            return chunker_class(max_tokens, overlap_chars=overlap, **kwargs)
        elif method == 'paragraphs':
            return chunker_class(max_tokens, overlap_paragraphs=overlap, **kwargs)
        elif method == 'code-blocks':
            return chunker_class(max_tokens, overlap_lines=overlap, **kwargs)
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
    result = [''] * original_length

    # Fill in the chunks
    for chunk in sorted_chunks:
        start = chunk.position.start
        end = chunk.position.end

        # Ensure we don't go out of bounds
        end = min(end, original_length)

        if start < original_length:
            chunk_chars = list(chunk.content[:end - start])
            result[start:end] = chunk_chars

    # Join and return
    return ''.join(result)


def find_chunk_containing_position(chunks: List[Chunk], position: int) -> Optional[Chunk]:
    """Find the chunk that contains a given character position"""
    for chunk in chunks:
        if chunk.position.start <= position < chunk.position.end:
            return chunk
    return None


def get_chunks_in_range(chunks: List[Chunk], start: int, end: int) -> List[Chunk]:
    """Get all chunks that overlap with a given range"""
    result = []
    for chunk in chunks:
        # Check if chunk overlaps with the range
        if (chunk.position.start < end and chunk.position.end > start):
            result.append(chunk)
    return result