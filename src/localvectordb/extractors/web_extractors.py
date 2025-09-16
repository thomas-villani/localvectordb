# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/extractors/web_extractors.py
"""
HTML and XML file extractors using BeautifulSoup4.
"""

import logging
from copy import copy
from typing import List, Optional

from localvectordb import MetadataField
from localvectordb.core import MetadataFieldType
from localvectordb.extractors import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class HTMLExtractor(BaseExtractor):
    """
    Extractor for HTML files using BeautifulSoup4.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.html', '.htm', '.xhtml']

    @property
    def supported_mimetypes(self) -> List[str]:
        return [
            'text/html',
            'application/xhtml+xml',
            'application/html'
        ]

    @property
    def required_packages(self) -> List[str]:
        return ['beautifulsoup4']

    @property
    def priority(self) -> int:
        return 20  # Higher than basic text extractor

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return {
            "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=False),
            "description": MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=False),
            "keywords": MetadataField(type=MetadataFieldType.JSON, indexed=False, required=False),
            "file_size_bytes": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "character_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "html_elements": MetadataField(type=MetadataFieldType.JSON, indexed=False, required=False),
        }

    def _check_availability(self) -> bool:
        try:
            from bs4 import BeautifulSoup
            return True
        except ImportError:
            return False

    def _extract_text_impl(
            self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs
            ) -> ExtractionResult:
        """Extract text from HTML files."""
        try:
            from bs4 import BeautifulSoup

            # Decode HTML content
            try:
                html_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try other encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        html_content = file_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return ExtractionResult(
                        text="",
                        success=False,
                        method='HTMLExtractor',
                        error="Could not decode HTML file with supported encodings"
                    )

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else None

            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content') if meta_description else None

            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            keywords = meta_keywords.get('content') if meta_keywords else None

            # Remove script and style elements
            for script in soup(["script", "style", "noscript"]):
                script.extract()

            # Extract text content
            text_parts = []

            # Add title if available
            if title_text:
                text_parts.append(f"Title: {title_text}")
                text_parts.append("")

            # Add meta description if available
            if description:
                text_parts.append(f"Description: {description}")
                text_parts.append("")

            # Extract structured content
            self._extract_structured_content(soup, text_parts)

            if not text_parts or (len(text_parts) <= 2 and title_text):
                return ExtractionResult(
                    text="",
                    success=False,
                    method='HTMLExtractor',
                    error="No meaningful text content found in HTML"
                )

            full_text = '\n'.join(text_parts)

            # Count various HTML elements
            element_counts = {
                'paragraphs': len(soup.find_all('p')),
                'headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                'links': len(soup.find_all('a')),
                'images': len(soup.find_all('img')),
                'tables': len(soup.find_all('table')),
                'lists': len(soup.find_all(['ul', 'ol']))
            }

            metadata = {
                'title': title_text,
                'description': description,
                'keywords': keywords,
                'file_size_bytes': len(file_content),
                'character_count': len(full_text),
                'html_elements': element_counts,
            }

            return ExtractionResult(
                text=full_text,
                success=True,
                method='HTMLExtractor',
                metadata=metadata
            )

        except Exception as e:
            return ExtractionResult(
                text="",
                success=False,
                method='HTMLExtractor',
                error=f"HTML extraction failed: {str(e)}"
            )

    def _extract_structured_content(self, soup, text_parts: List[str]):
        """Extract content in a structured way preserving hierarchy."""

        # Extract headings and content in order
        for element in soup.find_all(
                ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section', 'ul', 'ol', 'table']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Headings
                level = int(element.name[1])
                heading_text = element.get_text().strip()
                if heading_text:
                    prefix = '#' * level
                    text_parts.append(f"{prefix} {heading_text}")
                    text_parts.append("")

            elif element.name == 'p':
                # Paragraphs
                para_text = element.get_text().strip()
                if para_text:
                    text_parts.append(para_text)
                    text_parts.append("")

            elif element.name in ['ul', 'ol']:
                # Lists
                list_items = element.find_all('li', recursive=False)
                if list_items:
                    for i, li in enumerate(list_items, 1):
                        li_text = li.get_text().strip()
                        if li_text:
                            if element.name == 'ul':
                                text_parts.append(f"• {li_text}")
                            else:
                                text_parts.append(f"{i}. {li_text}")
                    text_parts.append("")

            elif element.name == 'table':
                # Tables
                self._extract_table_content(element, text_parts)

            elif element.name in ['div', 'article', 'section']:
                # Only extract if it contains direct text (not nested in other handled elements)
                direct_text = self._get_direct_text(element)
                if direct_text:
                    text_parts.append(direct_text)
                    text_parts.append("")

    @staticmethod
    def _extract_table_content(table, text_parts: List[str]):
        """Extract content from HTML tables."""
        rows = table.find_all('tr')
        if not rows:
            return

        text_parts.append("[Table]")

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_text = []
                for cell in cells:
                    cell_text = cell.get_text().strip()
                    row_text.append(cell_text)

                if any(cell for cell in row_text):  # At least one non-empty cell
                    text_parts.append(" | ".join(row_text))

        text_parts.append("")

    @staticmethod
    def _get_direct_text(element):
        """Get text that's directly in the element, not in nested handled elements."""
        # Remove nested elements that we handle separately
        temp_element = copy(element)
        for nested in temp_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']):
            nested.extract()

        text = temp_element.get_text().strip()
        return text if text and len(text) > 10 else None


class XMLExtractor(BaseExtractor):
    """
    Extractor for XML files using BeautifulSoup4.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.xml', '.xsl', '.xsd', '.svg', '.rss', '.atom']

    @property
    def supported_mimetypes(self) -> List[str]:
        return [
            'application/xml',
            'text/xml',
            'application/rss+xml',
            'application/atom+xml',
            'image/svg+xml'
        ]

    @property
    def required_packages(self) -> List[str]:
        return ['beautifulsoup4', 'lxml']

    @property
    def priority(self) -> int:
        return 20  # Higher than basic text extractor

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return {
            "xml_type": MetadataField(type=MetadataFieldType.TEXT, indexed=False, required=False),
            "item_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "character_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
        }

    def _check_availability(self) -> bool:
        try:
            import lxml
            from bs4 import BeautifulSoup
            return True
        except ImportError:
            try:
                from bs4 import BeautifulSoup
                # BeautifulSoup can work without lxml, but lxml is preferred for XML
                return True
            except ImportError:
                return False

    def _extract_text_impl(self, file_content: bytes, filename: str, mimetype: Optional[str]) -> ExtractionResult:
        """Extract text from XML files."""
        try:
            from bs4 import BeautifulSoup

            # Decode XML content
            try:
                xml_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try other encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        xml_content = file_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return ExtractionResult(
                        text="",
                        success=False,
                        method='XMLExtractor',
                        error="Could not decode XML file with supported encodings"
                    )

            # Choose parser based on availability
            try:
                import lxml
                parser = 'xml'  # lxml XML parser
            except ImportError:
                parser = 'html.parser'  # fallback to html.parser

            # Parse XML with BeautifulSoup
            soup = BeautifulSoup(xml_content, parser)

            # Detect XML type and extract accordingly
            root_tag = soup.find()
            xml_type = self._detect_xml_type(soup, root_tag)

            if xml_type == 'rss':
                return self._extract_rss_content(soup)
            elif xml_type == 'atom':
                return self._extract_atom_content(soup)
            elif xml_type == 'svg':
                return self._extract_svg_content(soup)
            else:
                return self._extract_generic_xml_content(soup, xml_content)

        except Exception as e:
            return ExtractionResult(
                text="",
                success=False,
                method='XMLExtractor',
                error=f"XML extraction failed: {str(e)}"
            )

    def _detect_xml_type(self, soup, root_tag):
        """Detect the type of XML document."""
        if not root_tag:
            return 'generic'

        root_name = root_tag.name.lower()

        if root_name == 'rss' or soup.find('channel'):
            return 'rss'
        elif root_name == 'feed' or soup.find('entry'):
            return 'atom'
        elif root_name == 'svg':
            return 'svg'
        else:
            return 'generic'

    def _extract_rss_content(self, soup):
        """Extract content from RSS feeds."""
        text_parts = []

        # Channel information
        channel = soup.find('channel')
        if channel:
            title = channel.find('title')
            if title:
                text_parts.append(f"RSS Feed: {title.get_text().strip()}")

            description = channel.find('description')
            if description:
                text_parts.append(f"Description: {description.get_text().strip()}")

            text_parts.append("")

        # Items
        items = soup.find_all('item')
        text_parts.append(f"Feed Items ({len(items)}):")
        text_parts.append("")

        for i, item in enumerate(items, 1):
            title = item.find('title')
            description = item.find('description')

            if title:
                text_parts.append(f"{i}. {title.get_text().strip()}")

            if description:
                desc_text = description.get_text().strip()
                # Clean HTML tags from description if present
                from bs4 import BeautifulSoup
                clean_desc = BeautifulSoup(desc_text, 'html.parser').get_text()
                text_parts.append(f"   {clean_desc}")

            text_parts.append("")

        full_text = '\n'.join(text_parts)

        metadata = {
            'xml_type': 'rss',
            'item_count': len(items),
            'character_count': len(full_text),
        }

        return ExtractionResult(
            text=full_text,
            success=True,
            method='XMLExtractor_RSS',
            metadata=metadata
        )

    def _extract_atom_content(self, soup):
        """Extract content from Atom feeds."""
        text_parts = []

        # Feed information
        feed_title = soup.find('title')
        if feed_title:
            text_parts.append(f"Atom Feed: {feed_title.get_text().strip()}")

        feed_subtitle = soup.find('subtitle')
        if feed_subtitle:
            text_parts.append(f"Subtitle: {feed_subtitle.get_text().strip()}")

        text_parts.append("")

        # Entries
        entries = soup.find_all('entry')
        text_parts.append(f"Feed Entries ({len(entries)}):")
        text_parts.append("")

        for i, entry in enumerate(entries, 1):
            title = entry.find('title')
            content = entry.find('content') or entry.find('summary')

            if title:
                text_parts.append(f"{i}. {title.get_text().strip()}")

            if content:
                content_text = content.get_text().strip()
                text_parts.append(f"   {content_text}")

            text_parts.append("")

        full_text = '\n'.join(text_parts)

        metadata = {
            'xml_type': 'atom',
            'entry_count': len(entries),
            'character_count': len(full_text),
        }

        return ExtractionResult(
            text=full_text,
            success=True,
            method='XMLExtractor_Atom',
            metadata=metadata
        )

    def _extract_svg_content(self, soup):
        """Extract content from SVG files."""
        text_parts = []

        # SVG metadata
        title = soup.find('title')
        if title:
            text_parts.append(f"SVG Title: {title.get_text().strip()}")

        desc = soup.find('desc')
        if desc:
            text_parts.append(f"Description: {desc.get_text().strip()}")

        # Extract text elements
        text_elements = soup.find_all('text')
        if text_elements:
            text_parts.append("")
            text_parts.append("Text Content:")
            for text_elem in text_elements:
                text_content = text_elem.get_text().strip()
                if text_content:
                    text_parts.append(f"- {text_content}")

        if not text_parts:
            return ExtractionResult(
                text="",
                success=False,
                method='XMLExtractor_SVG',
                error="No extractable text content found in SVG"
            )

        full_text = '\n'.join(text_parts)

        metadata = {
            'xml_type': 'svg',
            'text_elements': len(text_elements),
            'character_count': len(full_text),
        }

        return ExtractionResult(
            text=full_text,
            success=True,
            method='XMLExtractor_SVG',
            metadata=metadata
        )

    def _extract_generic_xml_content(self, soup, xml_content):
        """Extract content from generic XML files."""
        # For generic XML, extract all text content but preserve some structure
        text_parts = []

        # Get root element info
        root = soup.find()
        if root:
            text_parts.append(f"XML Document (Root: {root.name})")
            text_parts.append("")

        # Extract text from all elements, preserving hierarchy
        def extract_element_text(element, level=0):
            if not element.name:
                return

            # Get direct text content (not from children)
            direct_text = element.string
            if direct_text and direct_text.strip():
                indent = "  " * level
                text_parts.append(f"{indent}{element.name}: {direct_text.strip()}")
            elif element.name and level == 0:
                # For root elements without direct text, just show the element name
                text_parts.append(f"{element.name}:")

            # Process children
            for child in element.find_all(recursive=False):
                extract_element_text(child, level + 1)

        if root:
            extract_element_text(root)

        # Fallback: if structured extraction didn't work well, get all text
        if len(text_parts) < 3:
            all_text = soup.get_text()
            if all_text.strip():
                text_parts = ["XML Content:", "", all_text.strip()]

        if not text_parts or (len(text_parts) <= 2):
            return ExtractionResult(
                text="",
                success=False,
                method='XMLExtractor_Generic',
                error="No meaningful text content found in XML"
            )

        full_text = '\n'.join(text_parts)

        # Count elements
        all_elements = soup.find_all()
        element_counts = {}
        for elem in all_elements:
            if elem.name:
                element_counts[elem.name] = element_counts.get(elem.name, 0) + 1

        metadata = {
            'xml_type': 'generic',
            'root_element': root.name if root else None,
            'total_elements': len(all_elements),
            'element_types': len(element_counts),
            'element_counts': element_counts,
            'character_count': len(full_text),
        }

        return ExtractionResult(
            text=full_text,
            success=True,
            method='XMLExtractor_Generic',
            metadata=metadata
        )
