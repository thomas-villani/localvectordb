# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# tests/test_security_orderby.py
"""
Security tests for ORDER BY SQL injection prevention.

This test module specifically focuses on preventing SQL injection vulnerabilities
in the ORDER BY clause handling across the LocalVectorDB system.
"""

import pytest
import tempfile
from pathlib import Path

from localvectordb import VectorDB
from localvectordb._filters import FilterQueryBuilder
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.exceptions import DatabaseError


class TestOrderBySQLInjectionPrevention:
    """Test suite for ORDER BY SQL injection prevention."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_security.db"

    @pytest.fixture
    def test_schema(self):
        """Test metadata schema for validation."""
        return {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'rating': MetadataField(type=MetadataFieldType.REAL, indexed=True),
            'category': MetadataField(type=MetadataFieldType.TEXT),
            'year': MetadataField(type=MetadataFieldType.INTEGER),
        }

    @pytest.fixture
    def filter_builder(self, test_schema):
        """Create FilterQueryBuilder instance."""
        return FilterQueryBuilder(test_schema)

    @pytest.fixture
    def db_with_data(self, temp_db_path, test_schema):
        """Create database with test data."""
        db = VectorDB("test_security", temp_db_path.parent, metadata_schema=test_schema)
        
        # Add test documents
        docs = [
            "First test document",
            "Second test document", 
            "Third test document"
        ]
        metadata = [
            {'author': 'Alice', 'rating': 4.5, 'category': 'tech', 'year': 2023},
            {'author': 'Bob', 'rating': 3.8, 'category': 'science', 'year': 2022},
            {'author': 'Charlie', 'rating': 4.9, 'category': 'tech', 'year': 2024}
        ]
        
        db.upsert(documents=docs, metadata=metadata)
        return db

    def test_filter_builder_valid_order_by(self, filter_builder):
        """Test that valid ORDER BY clauses work correctly."""
        valid_columns = {'author', 'rating', 'category', 'year', 'id', 'created_at'}
        
        # Test valid cases
        valid_cases = [
            "author",
            "author ASC", 
            "author DESC",
            "rating ASC",
            "rating DESC",
            "created_at",
            "id DESC"
        ]
        
        for order_by in valid_cases:
            result = filter_builder.build_order_by_clause(order_by, valid_columns)
            assert "ORDER BY" in result
            assert "DESC" in result or "ASC" in result
            assert '"' in result  # Field should be quoted

    def test_filter_builder_sql_injection_attempts(self, filter_builder):
        """Test that SQL injection attempts are blocked."""
        valid_columns = {'author', 'rating', 'category', 'year', 'id', 'created_at'}
        
        # SQL injection attempts
        injection_attempts = [
            "author; DROP TABLE documents; --",
            "author UNION SELECT * FROM documents",
            "author' OR '1'='1",
            "author\"; DROP TABLE documents; --",
            "author; INSERT INTO documents VALUES (...)",
            "author/**/UNION/**/SELECT",
            "author'||CHR(39)||'1'||CHR(39)||'='||CHR(39)||'1",
            "author; EXEC xp_cmdshell('dir')",
            "(SELECT COUNT(*) FROM documents)",
            "author, (SELECT password FROM users LIMIT 1)",
            "author PROCEDURE ANALYSE()",
            "author AND SLEEP(5)",
            "author'; WAITFOR DELAY '00:00:05'--",
        ]
        
        for malicious_order_by in injection_attempts:
            with pytest.raises(DatabaseError, match=r"Invalid field name format|Invalid ORDER BY format"):
                filter_builder.build_order_by_clause(malicious_order_by, valid_columns)

    def test_filter_builder_invalid_field_names(self, filter_builder):
        """Test that invalid field names are rejected."""
        valid_columns = {'author', 'rating', 'category', 'year'}
        
        invalid_field_cases = [
            "nonexistent_field",
            "author, rating",  # Multiple fields not supported
            "author FROM documents",
            "author WHERE 1=1",
            "",
            "   ",
            "123invalid",
            "field-with-dashes",
            "field.with.dots",
            "field with spaces",
        ]
        
        for invalid_order_by in invalid_field_cases:
            with pytest.raises(DatabaseError):
                filter_builder.build_order_by_clause(invalid_order_by, valid_columns)

    def test_filter_builder_invalid_directions(self, filter_builder):
        """Test that invalid sort directions are rejected.""" 
        valid_columns = {'author', 'rating', 'category', 'year'}
        
        invalid_direction_cases = [
            "author ASCENDING",
            "author DESCENDING", 
            "author UP",
            "author DOWN",
            "author 1",
            "author ORDER",
            "author SELECT",
            "author UNION",
        ]
        
        for invalid_order_by in invalid_direction_cases:
            with pytest.raises(DatabaseError, match=r"ORDER BY direction must be ASC or DESC"):
                filter_builder.build_order_by_clause(invalid_order_by, valid_columns)

    def test_database_filter_sql_injection_prevention(self, db_with_data):
        """Test that database filter method prevents SQL injection."""
        
        # Valid ORDER BY should work
        result = db_with_data.filter(order_by="author ASC")
        assert len(result) == 3
        assert result[0].metadata['author'] <= result[1].metadata['author']
        
        # SQL injection attempts should fail
        injection_attempts = [
            "author; DROP TABLE documents; --",
            "author UNION SELECT * FROM sqlite_master",
            "author' OR '1'='1",
            "author\"; DROP TABLE documents; --"
        ]
        
        for malicious_order_by in injection_attempts:
            with pytest.raises(Exception):  # Should raise some form of exception
                db_with_data.filter(order_by=malicious_order_by)

    def test_async_filter_sql_injection_prevention(self, db_with_data):
        """Test that async database filter method prevents SQL injection."""
        import asyncio
        
        async def run_async_test():
            # Valid ORDER BY should work
            result = await db_with_data.filter_async(order_by="rating DESC")
            assert len(result) == 3
            assert result[0].metadata['rating'] >= result[1].metadata['rating']
            
            # SQL injection attempts should fail
            injection_attempts = [
                "rating; DROP TABLE documents; --",
                "rating UNION SELECT * FROM sqlite_master", 
                "rating' OR '1'='1",
                "rating\"; DROP TABLE documents; --"
            ]
            
            for malicious_order_by in injection_attempts:
                with pytest.raises(Exception):  # Should raise some form of exception
                    await db_with_data.filter_async(order_by=malicious_order_by)
        
        asyncio.run(run_async_test())

    def test_field_name_quoting(self, filter_builder):
        """Test that field names are properly quoted in SQL."""
        valid_columns = {'author', 'rating', 'id', 'created_at'}
        
        # Test that output contains quoted field names
        result = filter_builder.build_order_by_clause("author ASC", valid_columns)
        assert '"author"' in result
        assert 'ORDER BY "author" ASC' == result
        
        result = filter_builder.build_order_by_clause("created_at DESC", valid_columns)
        assert '"created_at"' in result
        assert 'ORDER BY "created_at" DESC' == result

    def test_reserved_column_validation(self, filter_builder):
        """Test that reserved columns are properly validated."""
        valid_columns = {'id', 'content', 'created_at', 'updated_at', 'content_hash'}
        
        # Reserved columns should be allowed
        reserved_cases = [
            "id",
            "created_at DESC",
            "updated_at ASC",
            "content_hash"
        ]
        
        for order_by in reserved_cases:
            result = filter_builder.build_order_by_clause(order_by, valid_columns)
            assert "ORDER BY" in result

    def test_empty_order_by_validation(self, filter_builder):
        """Test that empty ORDER BY clauses are rejected."""
        valid_columns = {'author', 'rating'}
        
        empty_cases = [
            "",
            "   "
        ]
        
        for empty_order_by in empty_cases:
            with pytest.raises(DatabaseError, match="ORDER BY clause cannot be empty"):
                filter_builder.build_order_by_clause(empty_order_by, valid_columns)
        
        # Test None separately since it would cause a different error
        with pytest.raises((DatabaseError, AttributeError)):
            filter_builder.build_order_by_clause(None, valid_columns)

    def test_case_sensitivity(self, filter_builder):
        """Test ORDER BY case sensitivity handling."""
        valid_columns = {'author', 'rating'}
        
        # Test case-insensitive directions
        case_variants = [
            "author asc",
            "author ASC", 
            "author Asc",
            "author desc",
            "author DESC",
            "author Desc"
        ]
        
        for order_by in case_variants:
            result = filter_builder.build_order_by_clause(order_by, valid_columns)
            assert "ORDER BY" in result
            assert ("ASC" in result) or ("DESC" in result)


class TestOrderBySecurityIntegration:
    """Integration tests for ORDER BY security across the full stack."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_integration.db"

    def test_end_to_end_sql_injection_prevention(self, temp_db_path):
        """Test end-to-end SQL injection prevention from API to database."""
        schema = {
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'priority': MetadataField(type=MetadataFieldType.INTEGER)
        }
        
        db = VectorDB("test_integration", temp_db_path.parent, metadata_schema=schema)
        
        # Add test data
        db.upsert(
            documents=["Test doc 1", "Test doc 2"], 
            metadata=[
                {'category': 'urgent', 'priority': 1},
                {'category': 'normal', 'priority': 2}
            ]
        )
        
        # Valid queries should work
        result = db.filter(order_by="priority DESC")
        assert len(result) == 2
        assert result[0].metadata['priority'] == 2  # Higher priority first
        
        # Injection attempts should be blocked
        dangerous_queries = [
            "priority; DELETE FROM documents;--",
            "priority UNION SELECT username,password FROM users--",
            "priority'; INSERT INTO documents SELECT * FROM sensitive_table;--",
            "priority\"; EXEC sp_configure 'show advanced options', 1;--"
        ]
        
        for dangerous_query in dangerous_queries:
            with pytest.raises(Exception):
                db.filter(order_by=dangerous_query)

    def test_schema_validation_consistency(self, temp_db_path):
        """Test that schema validation is consistent across sync/async."""
        schema = {
            'test_field': MetadataField(type=MetadataFieldType.TEXT)
        }
        
        db = VectorDB("test_consistency", temp_db_path.parent, metadata_schema=schema)
        db.upsert(documents=["Test"], metadata=[{'test_field': 'value'}])
        
        # Valid field should work in both sync and async
        sync_result = db.filter(order_by="test_field")
        assert len(sync_result) == 1
        
        import asyncio
        async def test_async():
            async_result = await db.filter_async(order_by="test_field")
            assert len(async_result) == 1
        
        asyncio.run(test_async())
        
        # Invalid field should fail in both sync and async
        with pytest.raises(Exception):
            db.filter(order_by="nonexistent_field")
            
        async def test_async_invalid():
            with pytest.raises(Exception):
                await db.filter_async(order_by="nonexistent_field")
        
        asyncio.run(test_async_invalid())