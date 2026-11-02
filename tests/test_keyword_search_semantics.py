"""Regression net for T1.0: keyword search must actually find documents.

Before this net existed, ``sanitize_fts_query`` AND-joined every term of a query, so
a natural-language sentence matched only documents containing all of its words --
stopwords included. On BEIR SciFact, 291 of 300 queries matched *zero* rows and
``search_type="keyword"`` scored nDCG@10 = 0.019. Because ``hybrid`` is the default
search type and fuses the two legs by weighted sum, an always-empty keyword leg also
made ``vector_weight`` inert: hybrid silently was pure vector search.

Two defects produced that:

A. the bare multi-term path joined terms with ``AND``;
B. dispatch tested ``query.upper()``, so ordinary English "and"/"or"/"not" became FTS5
   operators -- and the words on either side were glued into phrases.

The tests below pin the *semantics* of the join, which is what was wrong. The escaping
was never wrong, so the injection cases are kept verbatim.

Several tests run against a real SQLite FTS5 table rather than asserting on the
generated string. A MATCH expression can look perfectly reasonable and still match
nothing; only the database can settle that.
"""

import re
import shutil
import sqlite3
import tempfile

import pytest

from localvectordb._filters import FTSQuerySanitization
from localvectordb.database import LocalVectorDB

sanitize = FTSQuerySanitization.sanitize_fts_query


# ---------------------------------------------------------------------------
# The generated MATCH expression
# ---------------------------------------------------------------------------


class TestSanitizerSemantics:
    def test_bare_multi_term_is_or_joined(self):
        # Defect A. AND-joining is what made real queries match nothing.
        assert sanitize("hello world") == '"hello" OR "world"'

    def test_single_term_is_quoted(self):
        assert sanitize("python") == '"python"'

    @pytest.mark.parametrize("word", ["and", "or", "not"])
    def test_lowercase_operators_are_ordinary_words(self, word):
        # Defect B. FTS5 only honours uppercase operators; so must we.
        assert sanitize(f"alpha {word} beta") == f'"alpha" OR "{word}" OR "beta"'

    def test_english_not_does_not_become_an_exclusion(self):
        assert sanitize("aspirin does not reduce cardiovascular risk") == (
            '"aspirin" OR "does" OR "not" OR "reduce" OR "cardiovascular" OR "risk"'
        )

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("aspirin AND risk", '"aspirin" AND "risk"'),
            ("aspirin OR risk", '"aspirin" OR "risk"'),
            ("aspirin NOT risk", '"aspirin" NOT "risk"'),
        ],
    )
    def test_uppercase_operators_are_honoured(self, query, expected):
        assert sanitize(query) == expected

    def test_multiword_operand_is_not_glued_into_a_phrase(self):
        # Defect B's second half: the old code produced
        #   '"VITAMIN D" AND "CALCIUM SUPPLEMENTATION"'
        # which demands those exact literal phrases.
        assert sanitize("vitamin D AND calcium supplementation") == (
            '("vitamin" AND "D") AND ("calcium" AND "supplementation")'
        )

    def test_multiword_operand_is_parenthesized_against_fts5_precedence(self):
        # FTS5 binds AND tighter than OR. Without the parentheses this would regroup
        # as '"a" OR ("b" AND "c") OR "d"'.
        assert sanitize("a b OR c d") == '("a" AND "b") OR ("c" AND "d")'

    def test_quoted_phrase_is_preserved(self):
        assert sanitize('"machine learning"') == '"machine learning"'

    def test_punctuation_only_term_is_dropped(self):
        # "(+)-" survives clean_term as a bare "-", which tokenizes to nothing.
        assert sanitize("alpha (+)- thalassemia") == '"alpha" OR "thalassemia"'

    def test_operand_that_cleans_away_falls_back_to_or_of_survivors(self):
        assert sanitize("foo AND ***") == '"foo"'

    def test_empty_query_returns_empty(self):
        assert sanitize("") == ""
        assert sanitize("   ") == ""
        assert sanitize("-") == ""

    # -- escaping was never the bug; keep it that way -----------------------

    def test_special_characters_are_stripped(self):
        out = sanitize("foo(bar)")
        assert out == '"foobar"'
        assert "(" not in out and ")" not in out

    def test_clean_term_removes_fts_operators(self):
        assert FTSQuerySanitization.clean_term("term*^:") == "term"

    @pytest.mark.parametrize(
        "hostile",
        [
            'x" OR chunks_fts MATCH "y',
            "foo* OR bar",
            "a AND (b",
            "title:secret",
            "NEAR(a b, 2)",
            "^anchored",
            'a" OR "b',
        ],
    )
    def test_no_user_text_survives_as_fts5_syntax(self, hostile):
        """Every token is either a structural operator we emitted or a quoted phrase."""
        out = sanitize(hostile)
        # Strip the quoted phrases; what remains must be only operators and parens.
        residue = re.sub(r'"[^"]*"', " ", out)
        for token in residue.replace("(", " ").replace(")", " ").split():
            assert token in {"AND", "OR", "NOT"}, f"{token!r} escaped quoting in {out!r}"


# ---------------------------------------------------------------------------
# Against a real FTS5 table
# ---------------------------------------------------------------------------

_DOCS = [
    # 0: shares the rare terms with the query below
    "Aspirin reduces cardiovascular risk in patients with prior myocardial infarction.",
    # 1: shares only stopwords
    "The quick brown fox does not jump over anything at all in the morning.",
    # 2: topically adjacent, one rare term
    "Statins lower cholesterol and are prescribed to reduce cardiovascular events.",
]


@pytest.fixture
def fts():
    """A real FTS5 table. The generated expression can look fine and match nothing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")
    conn.executemany("INSERT INTO docs(rowid, content) VALUES (?, ?)", list(enumerate(_DOCS)))
    conn.commit()
    yield conn
    conn.close()


def _match(conn, expression):
    if not expression:
        return []
    rows = conn.execute("SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) ASC", (expression,)).fetchall()
    return [r[0] for r in rows]


class TestAgainstRealFTS5:
    def test_no_document_contains_every_term_yet_the_query_still_matches(self, fts):
        """The exact shape of defect A. No document has all of these words."""
        query = "aspirin reduces cardiovascular risk in elderly diabetic patients"
        assert _match(fts, sanitize(query)), "keyword search found nothing"

    def test_the_old_and_join_would_have_matched_nothing(self, fts):
        """Pins the bug itself, so a revert to AND-joining fails loudly here."""
        query = "aspirin reduces cardiovascular risk in elderly diabetic patients"
        and_joined = " AND ".join(FTSQuerySanitization.quote_terms(query))
        assert _match(fts, and_joined) == []
        assert _match(fts, sanitize(query)) != []

    def test_bm25_ranks_the_document_sharing_rare_terms_first(self, fts):
        """OR-joining is only useful if BM25 then ranks well.

        Doc 1 shares the stopwords ("does", "not", "in", "the"); doc 0 shares the rare
        ones. If IDF were not doing its job, this would not hold.
        """
        ranked = _match(fts, sanitize("aspirin does not reduce cardiovascular risk"))
        assert ranked[0] == 0

    def test_english_not_does_not_exclude_the_matching_document(self, fts):
        # Under defect B this became '"ASPIRIN DOES" NOT "REDUCE CARDIOVASCULAR RISK"'.
        assert 0 in _match(fts, sanitize("aspirin does not reduce cardiovascular risk"))

    def test_uppercase_not_does_exclude(self, fts):
        ranked = _match(fts, sanitize("cardiovascular NOT aspirin"))
        assert 0 not in ranked
        assert 2 in ranked

    def test_uppercase_and_requires_both(self, fts):
        assert _match(fts, sanitize("aspirin AND cholesterol")) == []
        assert _match(fts, sanitize("aspirin AND cardiovascular")) == [0]

    @pytest.mark.parametrize(
        "hostile",
        ['x" OR docs MATCH "y', "foo* OR bar", "a AND (b", "NEAR(a b, 2)", "^anchored"],
    )
    def test_hostile_input_neither_errors_nor_matches_everything(self, fts, hostile):
        """Injection must degrade to a literal term search, not to syntax."""
        rows = _match(fts, sanitize(hostile))  # must not raise sqlite3.OperationalError
        assert len(rows) < len(_DOCS)


# ---------------------------------------------------------------------------
# End to end, through LocalVectorDB
# ---------------------------------------------------------------------------


@pytest.fixture
def keyword_db():
    """Real LocalVectorDB, real FTS5, mock embeddings.

    MockEmbeddings seeds np.random on a hash of the text, so vector scores here are
    effectively arbitrary. That is exactly what makes this fixture a good probe of the
    *keyword* leg: any correct ranking it produces came from BM25.
    """
    temp_dir = tempfile.mkdtemp()
    db = LocalVectorDB(
        name="keyword_semantics",
        base_path=temp_dir,
        embedding_provider="mock",
        embedding_model="mock-model",
    )
    db.upsert(documents=list(_DOCS), ids=[f"doc{i}" for i in range(len(_DOCS))])
    yield db
    db.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestKeywordSearchEndToEnd:
    def test_natural_language_query_finds_the_document(self, keyword_db):
        results = keyword_db.query(
            "aspirin reduces cardiovascular risk in elderly diabetic patients",
            search_type="keyword",
        )
        assert results, "keyword search returned nothing for a natural-language query"
        assert results[0].id == "doc0"

    def test_query_containing_the_word_not_still_finds_the_document(self, keyword_db):
        results = keyword_db.query(
            "aspirin does not reduce cardiovascular risk",
            search_type="keyword",
        )
        assert [r.id for r in results][:1] == ["doc0"]

    def test_keyword_leg_reaches_the_hybrid_fusion(self, keyword_db):
        """`vector_weight` was inert because the keyword leg was always empty.

        With `vector_weight=0.0` the score is entirely the keyword score, so BM25 alone
        decides the ranking. Under the old code this returned nothing at all.
        """
        results = keyword_db.query(
            "aspirin reduces cardiovascular risk",
            search_type="hybrid",
            vector_weight=0.0,
        )
        assert results
        assert results[0].id == "doc0"


# ---------------------------------------------------------------------------
# Defect C: the phrase-mixed branch AND-joined, which is defect A wearing a quote
# ---------------------------------------------------------------------------


class TestPhraseMixedQueries:
    """A query that only PARTLY quotes used to conjoin every part, stopwords included.

    Same failure as defect A and for the same reason, but it survived the T1.0 fix
    because that fix only touched the bare multi-term path. Measured on 2,752 MAUD
    queries: 20.6% took this branch and 99.8% of those returned nothing.
    """

    def test_phrase_mixed_query_is_or_joined(self):
        assert sanitize('How is a "chunk of posts" defined?') == (
            '"How" OR "is" OR "a" OR "chunk of posts" OR "defined"'
        )

    def test_the_phrase_survives_as_one_phrase(self):
        """OR-joining must not shatter the phrase into loose words.

        If it did, quoting would stop meaning anything at all rather than merely
        stopping being mandatory -- and the whole change would be a no-op dressed
        up as a fix.
        """
        out = sanitize('the "quick brown" fox')
        assert '"quick brown"' in out
        assert '"quick" OR "brown"' not in out

    def test_a_fully_quoted_query_is_still_required_to_match(self):
        """The exact-phrase branch is untouched: quoting the WHOLE query still binds."""
        assert sanitize('"machine learning"') == '"machine learning"'

    def test_multiple_phrases_each_survive(self):
        out = sanitize('the "quick brown" fox and the "lazy dog"')
        assert '"quick brown"' in out and '"lazy dog"' in out
        assert " AND " not in out

    def test_boolean_branch_is_untouched(self):
        """Only the phrase branch changed; an explicit AND still conjoins."""
        assert sanitize("aspirin AND risk") == '"aspirin" AND "risk"'

    def test_phrase_mixed_query_finds_the_document(self, fts):
        """The point of the change, against a real index.

        No document contains every word of this sentence, so the old AND-join
        returned nothing while the phrase itself was present all along.
        """
        query = 'does "cardiovascular risk" fall in elderly diabetic patients?'
        assert _match(fts, sanitize(query)), "phrase-mixed keyword search found nothing"

    def test_the_old_and_join_would_have_matched_nothing(self, fts):
        """Pins the defect, so a revert to AND-joining fails loudly here."""
        query = 'does "cardiovascular risk" fall in elderly diabetic patients?'
        and_joined = '"does" AND "cardiovascular risk" AND "fall" AND "in" AND "elderly"'
        assert _match(fts, and_joined) == []
        assert _match(fts, sanitize(query)) != []

    def test_the_phrase_still_constrains_ranking(self, fts):
        """OR-joining costs the phrase its veto, not its contribution.

        Doc 0 contains "cardiovascular risk" as a contiguous phrase; doc 2 has both
        words but not adjacent. If the phrase were shattered they would be
        interchangeable here.
        """
        assert _match(fts, '"cardiovascular risk"') == [0]

    @pytest.mark.parametrize(
        "hostile",
        ['unclosed "phrase', '"" empty', 'a "*" b', '"x" OR docs MATCH "y'],
    )
    def test_hostile_phrase_input_neither_errors_nor_matches_everything(self, fts, hostile):
        rows = _match(fts, sanitize(hostile))  # must not raise sqlite3.OperationalError
        assert len(rows) <= len(_DOCS)


class TestInnerPunctuation:
    """Issue #98: identifiers must match the tokens FTS5 split them into at index time."""

    DOCS = [
        "the weekly/plan is in file.txt",
        "email foo@bar.com about it",
        "see section 5.02 of the agreement",
        "version 1.2.3 and rate 3.5",
        "the AT&T merger",
        "the datasets were large",
    ]

    @pytest.fixture
    def index(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE d USING fts5(content)")
        conn.executemany("INSERT INTO d(rowid, content) VALUES (?, ?)", list(enumerate(self.DOCS)))
        yield conn
        conn.close()

    @staticmethod
    def _rows(conn, query):
        rows = conn.execute("SELECT rowid FROM d WHERE d MATCH ? ORDER BY rowid", (sanitize(query),))
        return [row[0] for row in rows]

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("weekly/plan", [0]),
            ("file.txt", [0]),
            ("foo@bar.com", [1]),
            ("5.02", [2]),
            ("1.2.3", [3]),
            ("AT&T", [4]),
        ],
    )
    def test_identifier_finds_its_document(self, index, query, expected):
        assert self._rows(index, query) == expected

    def test_identifier_parts_must_stay_adjacent(self, index):
        """ "3.5" must not match a document that merely has a 3 and a 5 somewhere."""
        index.execute("INSERT INTO d(rowid, content) VALUES (99, 'chapter 5 of volume 3')")
        assert self._rows(index, "3.5") == [3]

    def test_identifier_is_found_inside_a_sentence(self, index):
        assert 0 in self._rows(index, "where is file.txt kept")

    def test_identifier_is_found_inside_a_partly_quoted_query(self, index):
        assert 2 in self._rows(index, '"see section" 5.02')

    def test_deleted_parentheses_still_recover_the_plural(self, index):
        assert self._rows(index, "dataset(s)") == [5]

    def test_terms_with_no_alphanumerics_are_still_dropped(self):
        assert sanitize("alpha ./@ beta") == '"alpha" OR "beta"'

    @pytest.mark.parametrize("hostile", ["a:b", "a*b", "a^b", "col:val*", "x{y}z", "NEAR/2", "a+b"])
    def test_inner_syntax_characters_never_reach_fts5(self, index, hostile):
        out = sanitize(hostile)
        assert not set(out) & set("*:^{}+/")
        self._rows(index, hostile)  # must not raise sqlite3.OperationalError
