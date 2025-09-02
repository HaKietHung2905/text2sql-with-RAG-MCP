#!/usr/bin/env python3
"""
MCP Server implementations for Text2SQL
Provides different context servers for enhanced SQL generation

File: mcp/servers.py
"""

import asyncio
import json
import sqlite3
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# MCP Server imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool, 
        TextContent, 
        ImageContent, 
        EmbeddedResource,
        Resource,
        Prompt,
        GetPromptResult,
        CallToolResult
    )
    MCP_SERVER_AVAILABLE = True
except ImportError:
    MCP_SERVER_AVAILABLE = False
    print("⚠️  MCP Server not available. Install with: pip install mcp")

# ================================
# Database Schema Server
# ================================

class DatabaseSchemaServer:
    """MCP Server for database schema information"""
    
    def __init__(self):
        self.server = Server("database-schema")
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools for database schema operations"""
        
        @self.server.call_tool()
        async def get_schema(database_path: str) -> List[TextContent]:
            """Get complete database schema information"""
            try:
                if not os.path.exists(database_path):
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"Database not found: {database_path}"})
                    )]
                
                schema_info = self._extract_schema(database_path)
                return [TextContent(
                    type="text", 
                    text=json.dumps(schema_info, indent=2)
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_relationships(database_path: str, table_name: str) -> List[TextContent]:
            """Get table relationships and foreign keys"""
            try:
                relationships = self._get_table_relationships(database_path, table_name)
                return [TextContent(
                    type="text",
                    text=json.dumps(relationships, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        

        
        @self.server.call_tool()
        async def analyze_column_types(database_path: str, table_name: str) -> List[TextContent]:
            """Analyze column types and constraints"""
            try:
                analysis = self._analyze_column_types(database_path, table_name)
                return [TextContent(
                    type="text",
                    text=json.dumps(analysis, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
    
    def _extract_schema(self, database_path: str) -> Dict[str, Any]:
        """Extract complete schema information"""
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        try:
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema = {
                "database_path": database_path,
                "database_name": Path(database_path).stem,
                "tables": {},
                "relationships": {},
                "indexes": {},
                "total_tables": len(tables)
            }
            
            for table in tables:
                # Get table info
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                table_info = {
                    "columns": {},
                    "primary_keys": [],
                    "foreign_keys": [],
                    "column_count": len(columns),
                    "constraints": []
                }
                
                for col in columns:
                    col_name = col[1]
                    table_info["columns"][col_name] = {
                        "type": col[2],
                        "nullable": not col[3],
                        "default": col[4],
                        "primary_key": col[5] == 1,
                        "position": col[0]
                    }
                    
                    if col[5] == 1:  # Primary key
                        table_info["primary_keys"].append(col_name)
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                fkeys = cursor.fetchall()
                for fkey in fkeys:
                    table_info["foreign_keys"].append({
                        "id": fkey[0],
                        "seq": fkey[1],
                        "table": fkey[2],
                        "from": fkey[3],
                        "to": fkey[4],
                        "on_update": fkey[5],
                        "on_delete": fkey[6],
                        "match": fkey[7]
                    })
                
                # Get indexes
                cursor.execute(f"PRAGMA index_list({table})")
                indexes = cursor.fetchall()
                table_indexes = []
                for idx in indexes:
                    cursor.execute(f"PRAGMA index_info({idx[1]})")
                    idx_info = cursor.fetchall()
                    table_indexes.append({
                        "name": idx[1],
                        "unique": bool(idx[2]),
                        "origin": idx[3],
                        "partial": bool(idx[4]),
                        "columns": [col[2] for col in idx_info]
                    })
                
                schema["indexes"][table] = table_indexes
                schema["tables"][table] = table_info
            
            return schema
            
        finally:
            conn.close()
    
    def _get_table_relationships(self, database_path: str, table_name: str) -> Dict[str, Any]:
        """Get relationships for a specific table"""
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        try:
            relationships = {
                "table": table_name,
                "foreign_keys": [],
                "referenced_by": [],
                "self_references": []
            }
            
            # Get foreign keys from this table
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fkeys = cursor.fetchall()
            for fkey in fkeys:
                fk_info = {
                    "column": fkey[3],
                    "references_table": fkey[2],
                    "references_column": fkey[4],
                    "on_update": fkey[5],
                    "on_delete": fkey[6]
                }
                
                if fkey[2] == table_name:  # Self-reference
                    relationships["self_references"].append(fk_info)
                else:
                    relationships["foreign_keys"].append(fk_info)
            
            # Find tables that reference this table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            for other_table in all_tables:
                if other_table == table_name:
                    continue
                
                cursor.execute(f"PRAGMA foreign_key_list({other_table})")
                other_fkeys = cursor.fetchall()
                for fkey in other_fkeys:
                    if fkey[2] == table_name:  # References our table
                        relationships["referenced_by"].append({
                            "table": other_table,
                            "column": fkey[3],
                            "references_column": fkey[4],
                            "on_update": fkey[5],
                            "on_delete": fkey[6]
                        })
            
            return relationships
            
        finally:
            conn.close()
    

    
    def _analyze_column_types(self, database_path: str, table_name: str) -> Dict[str, Any]:
        """Analyze column types and constraints"""
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        try:
            analysis = {
                "table": table_name,
                "columns": {}
            }
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                
                column_analysis = {
                    "declared_type": col_type,
                    "nullable": not col[3],
                    "default_value": col[4],
                    "primary_key": col[5] == 1
                }
                
                analysis["columns"][col_name] = column_analysis
            
            return analysis
            
        finally:
            conn.close()
    
# ================================
# SQL Documentation Server
# ================================

class SQLDocumentationServer:
    """MCP Server for SQL documentation and examples"""
    
    def __init__(self):
        self.server = Server("sql-documentation")
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools for SQL documentation operations"""
        
        @self.server.call_tool()
        async def get_syntax_help(sql_keyword: str) -> List[TextContent]:
            """Get syntax help for SQL keywords"""
            try:
                help_info = self._get_syntax_help(sql_keyword)
                return [TextContent(
                    type="text",
                    text=json.dumps(help_info, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def validate_sql(sql_query: str) -> List[TextContent]:
            """Validate SQL syntax and provide suggestions"""
            try:
                validation = self._validate_sql(sql_query)
                return [TextContent(
                    type="text",
                    text=json.dumps(validation, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_best_practices(topic: str) -> List[TextContent]:
            """Get SQL best practices for specific topics"""
            try:
                practices = self._get_best_practices(topic)
                return [TextContent(
                    type="text",
                    text=json.dumps(practices, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def explain_sql(sql_query: str) -> List[TextContent]:
            """Explain what a SQL query does"""
            try:
                explanation = self._explain_sql(sql_query)
                return [TextContent(
                    type="text",
                    text=json.dumps(explanation, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
    
    
    def _get_syntax_help(self, sql_keyword: str) -> Dict[str, Any]:
        """Get syntax help for SQL keywords"""
        keyword = sql_keyword.lower().replace('_', ' ')
        help_info = self.syntax_db.get(keyword)
        
        if not help_info:
            return {"error": f"No help available for '{sql_keyword}'"}
        
        return {
            "keyword": sql_keyword,
            "help": help_info
        }
    
    def _get_best_practices(self, topic: str) -> Dict[str, Any]:
        """Get best practices for specific topics"""
        practices = self.best_practices_db.get(topic.lower(), [])
        
        if not practices:
            available_topics = list(self.best_practices_db.keys())
            return {
                "error": f"No practices found for '{topic}'",
                "available_topics": available_topics
            }
        
        return {
            "topic": topic,
            "practices": practices,
            "practice_count": len(practices)
        }
    
    def _validate_sql(self, sql_query: str) -> Dict[str, Any]:
        """Basic SQL validation and suggestions"""
        issues = []
        suggestions = []
        warnings = []
        
        sql_upper = sql_query.upper().strip()
        
        # Basic syntax checks
        if not sql_upper.startswith('SELECT'):
            issues.append("Query should start with SELECT")
        
        if 'FROM' not in sql_upper:
            issues.append("Missing FROM clause")
        
        # Check for common syntax issues
        if sql_query.count('(') != sql_query.count(')'):
            issues.append("Unmatched parentheses")
        
        if sql_query.count("'") % 2 != 0:
            issues.append("Unmatched single quotes")
        
        if sql_query.count('"') % 2 != 0:
            issues.append("Unmatched double quotes")
        
        # Check for missing semicolon (warning, not error)
        if not sql_query.strip().endswith(';'):
            warnings.append("Consider ending SQL statements with semicolon")
        
        # Performance suggestions
        if '*' in sql_query and 'COUNT(*)' not in sql_upper:
            suggestions.append("Consider selecting specific columns instead of *")
        
        if 'WHERE' not in sql_upper and 'SELECT' in sql_upper:
            suggestions.append("Consider adding WHERE clause for filtering")
        
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            suggestions.append("Consider adding LIMIT clause with ORDER BY")
        
        # Advanced checks
        if re.search(r'SELECT\s+\*\s+FROM.*JOIN', sql_upper):
            suggestions.append("Specify columns when using JOINs to improve performance")
        
        if 'GROUP BY' in sql_upper and 'HAVING' not in sql_upper:
            suggestions.append("Consider using HAVING clause to filter grouped results")
        
        return {
            "sql_query": sql_query,
            "is_valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "warnings": warnings
        }
    
    def _explain_sql(self, sql_query: str) -> Dict[str, Any]:
        """Explain what a SQL query does"""
        explanation = {
            "query": sql_query,
            "explanation": {
                "overview": "",
                "clauses": [],
                "operations": [],
                "estimated_complexity": ""
            }
        }
        
        sql_upper = sql_query.upper()
        
        # Basic overview
        if sql_upper.startswith('SELECT'):
            explanation["explanation"]["overview"] = "This is a SELECT query that retrieves data from one or more tables."
        
        # Analyze clauses
        clauses = []
        
        if 'SELECT' in sql_upper:
            if '*' in sql_query:
                clauses.append("SELECT *: Retrieves all columns from the specified table(s)")
            else:
                clauses.append("SELECT: Specifies which columns to retrieve")
        
        if 'FROM' in sql_upper:
            clauses.append("FROM: Specifies the source table(s) for the data")
        
        if 'WHERE' in sql_upper:
            clauses.append("WHERE: Filters rows based on specified conditions")
        
        if 'JOIN' in sql_upper:
            join_type = "INNER"
            if 'LEFT JOIN' in sql_upper:
                join_type = "LEFT"
            elif 'RIGHT JOIN' in sql_upper:
                join_type = "RIGHT"
            elif 'FULL' in sql_upper and 'JOIN' in sql_upper:
                join_type = "FULL OUTER"
            
            clauses.append(f"{join_type} JOIN: Combines data from multiple tables")
        
        if 'GROUP BY' in sql_upper:
            clauses.append("GROUP BY: Groups rows with same values in specified columns")
        
        if 'HAVING' in sql_upper:
            clauses.append("HAVING: Filters groups based on aggregate conditions")
        
        if 'ORDER BY' in sql_upper:
            clauses.append("ORDER BY: Sorts the result set")
        
        if 'LIMIT' in sql_upper:
            clauses.append("LIMIT: Restricts the number of rows returned")
        
        explanation["explanation"]["clauses"] = clauses
        
        # Identify operations
        operations = []
        
        if re.search(r'COUNT\s*\(', sql_upper):
            operations.append("COUNT: Counts the number of rows")
        
        if re.search(r'SUM\s*\(', sql_upper):
            operations.append("SUM: Calculates the total of numeric values")
        
        if re.search(r'AVG\s*\(', sql_upper):
            operations.append("AVG: Calculates the average of numeric values")
        
        if re.search(r'MAX\s*\(', sql_upper):
            operations.append("MAX: Finds the maximum value")
        
        if re.search(r'MIN\s*\(', sql_upper):
            operations.append("MIN: Finds the minimum value")
        
        explanation["explanation"]["operations"] = operations
        
        return explanation
    
    def _calculate_complexity(self, sql_query: str) -> int:
        """Calculate query complexity score"""
        score = 0
        sql_upper = sql_query.upper()
        
        # Basic clauses
        if 'WHERE' in sql_upper:
            score += 1
        if 'JOIN' in sql_upper:
            score += 2
        if 'GROUP BY' in sql_upper:
            score += 1
        if 'HAVING' in sql_upper:
            score += 1
        if 'ORDER BY' in sql_upper:
            score += 1
        
        # Advanced features
        if 'UNION' in sql_upper or 'INTERSECT' in sql_upper or 'EXCEPT' in sql_upper:
            score += 3
        
        # Subqueries
        subquery_count = sql_query.count('(') - sql_query.count('COUNT(') - sql_query.count('SUM(') - sql_query.count('AVG(') - sql_query.count('MAX(') - sql_query.count('MIN(')
        score += subquery_count
        
        # Aggregate functions
        aggregates = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']
        for agg in aggregates:
            score += sql_upper.count(agg)
        
        return score

# ================================
# Query History Server
# ================================

class QueryHistoryServer:
    """MCP Server for query history and pattern matching"""
    
    def __init__(self):
        self.server = Server("query-history")
        self.history_file = Path("data/mcp_data/query_history.json")
        self.query_history = self._load_history()
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup MCP tools for query history operations"""
        
        @self.server.call_tool()
        async def search_queries(keywords: List[str], limit: int = 10) -> List[TextContent]:
            """Search historical queries by keywords"""
            try:
                results = self._search_by_keywords(keywords, limit)
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def add_query(question: str, sql: str, db_name: str, success: bool = True, 
                          execution_time: float = None, confidence: float = None) -> List[TextContent]:
            """Add a query to history"""
            try:
                result = self._add_to_history(question, sql, db_name, success, execution_time, confidence)
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_similar_queries(question: str, limit: int = 5) -> List[TextContent]:
            """Find similar queries based on question patterns"""
            try:
                similar = self._find_similar_queries(question, limit)
                return [TextContent(
                    type="text",
                    text=json.dumps(similar, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_query_patterns(pattern_type: str = "all") -> List[TextContent]:
            """Get common query patterns from history"""
            try:
                patterns = self._extract_patterns(pattern_type)
                return [TextContent(
                    type="text",
                    text=json.dumps(patterns, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_performance_stats(db_name: str = None) -> List[TextContent]:
            """Get performance statistics for queries"""
            try:
                stats = self._get_performance_stats(db_name)
                return [TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
        
        @self.server.call_tool()
        async def get_recent_queries(limit: int = 20) -> List[TextContent]:
            """Get recent queries from history"""
            try:
                recent = self._get_recent_queries(limit)
                return [TextContent(
                    type="text",
                    text=json.dumps(recent, indent=2)
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]
    
    def _load_history(self) -> List[Dict]:
        """Load query history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
        
        # Return default history if file doesn't exist
        return self._get_default_history()
    
    def _get_default_history(self) -> List[Dict]:
        """Get default query history"""
        return [
            {
                "question": "How many students are there?",
                "sql": "SELECT COUNT(*) FROM student",
                "db_name": "university",
                "success": True,
                "timestamp": "2024-01-01T10:00:00",
                "keywords": ["count", "students"],
                "execution_time": 0.023,
                "confidence": 0.95
            },
            {
                "question": "List all teachers",
                "sql": "SELECT * FROM teacher",
                "db_name": "university", 
                "success": True,
                "timestamp": "2024-01-01T10:05:00",
                "keywords": ["list", "teachers"],
                "execution_time": 0.015,
                "confidence": 0.90
            },
            {
                "question": "Show student names with their courses",
                "sql": "SELECT s.name, c.course_name FROM student s JOIN enrollment e ON s.id = e.student_id JOIN course c ON e.course_id = c.id",
                "db_name": "university",
                "success": True,
                "timestamp": "2024-01-01T10:10:00",
                "keywords": ["students", "courses", "join"],
                "execution_time": 0.087,
                "confidence": 0.85
            }
        ]
    
    def _save_history(self):
        """Save query history to file"""
        try:
            # Ensure directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.history_file, 'w') as f:
                json.dump(self.query_history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def _search_by_keywords(self, keywords: List[str], limit: int) -> List[Dict]:
        """Search queries by keywords"""
        results = []
        keywords_lower = [k.lower() for k in keywords]
        
        for query in self.query_history:
            score = 0
            query_text = (query['question'] + ' ' + query['sql']).lower()
            
            for keyword in keywords_lower:
                # Exact keyword match in text
                if keyword in query_text:
                    score += 1
                
                # Higher score for explicit keywords
                if keyword in query.get('keywords', []):
                    score += 2
                
                # Partial matches
                for stored_keyword in query.get('keywords', []):
                    if keyword in stored_keyword or stored_keyword in keyword:
                        score += 0.5
            
            if score > 0:
                result_entry = {
                    **query,
                    "relevance_score": score,
                    "match_reasons": self._get_match_reasons(keywords_lower, query)
                }
                results.append(result_entry)
        
        # Sort by relevance score and recency
        results.sort(key=lambda x: (x['relevance_score'], x['timestamp']), reverse=True)
        return results[:limit]
    
    def _get_match_reasons(self, keywords: List[str], query: Dict) -> List[str]:
        """Get reasons why a query matched the search"""
        reasons = []
        query_text = (query['question'] + ' ' + query['sql']).lower()
        
        for keyword in keywords:
            if keyword in query['question'].lower():
                reasons.append(f"'{keyword}' found in question")
            elif keyword in query['sql'].lower():
                reasons.append(f"'{keyword}' found in SQL")
            elif keyword in query.get('keywords', []):
                reasons.append(f"'{keyword}' is a tagged keyword")
        
        return reasons
    
    def _add_to_history(self, question: str, sql: str, db_name: str, success: bool, 
                       execution_time: float = None, confidence: float = None) -> Dict:
        """Add a new query to history"""
        keywords = self._extract_keywords_from_question(question)
        
        new_entry = {
            "question": question,
            "sql": sql,
            "db_name": db_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "keywords": keywords,
            "execution_time": execution_time,
            "confidence": confidence,
            "query_type": self._classify_query_type(sql)
        }
        
        self.query_history.append(new_entry)
        
        # Keep only recent entries (last 500)
        if len(self.query_history) > 500:
            self.query_history = self.query_history[-500:]
        
        self._save_history()
        
        return {"status": "added", "entry": new_entry, "total_entries": len(self.query_history)}
    
    def _find_similar_queries(self, question: str, limit: int) -> List[Dict]:
        """Find similar queries based on question patterns"""
        question_lower = question.lower()
        similar = []
        
        for query in self.query_history:
            # Text similarity
            text_similarity = self._calculate_text_similarity(question_lower, query['question'].lower())
            
            # Keyword overlap
            question_keywords = set(self._extract_keywords_from_question(question))
            query_keywords = set(query.get('keywords', []))
            keyword_similarity = len(question_keywords & query_keywords) / max(len(question_keywords | query_keywords), 1)
            
            # Pattern similarity
            pattern_similarity = self._calculate_pattern_similarity(question, query['question'])
            
            # Combined similarity score
            combined_similarity = (text_similarity * 0.4 + keyword_similarity * 0.4 + pattern_similarity * 0.2)
            
            if combined_similarity > 0.3:  # Threshold for similarity
                similar.append({
                    **query,
                    "similarity_score": combined_similarity,
                    "text_similarity": text_similarity,
                    "keyword_similarity": keyword_similarity,
                    "pattern_similarity": pattern_similarity
                })
        
        similar.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar[:limit]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_pattern_similarity(self, question1: str, question2: str) -> float:
        """Calculate pattern similarity based on question structure"""
        patterns1 = self._extract_question_patterns(question1)
        patterns2 = self._extract_question_patterns(question2)
        
        common_patterns = patterns1 & patterns2
        total_patterns = patterns1 | patterns2
        
        if len(total_patterns) == 0:
            return 0.0
        
        return len(common_patterns) / len(total_patterns)
    
    def _extract_question_patterns(self, question: str) -> set:
        """Extract structural patterns from questions"""
        question_lower = question.lower()
        patterns = set()
        
        # Question types
        if question_lower.startswith('how many'):
            patterns.add('count_question')
        elif question_lower.startswith('what is'):
            patterns.add('what_question')
        elif question_lower.startswith('show') or question_lower.startswith('list'):
            patterns.add('list_question')
        elif question_lower.startswith('find'):
            patterns.add('find_question')
        
        # Keywords
        if 'average' in question_lower or 'mean' in question_lower:
            patterns.add('average_query')
        if 'total' in question_lower or 'sum' in question_lower:
            patterns.add('sum_query')
        if 'maximum' in question_lower or 'highest' in question_lower:
            patterns.add('max_query')
        if 'minimum' in question_lower or 'lowest' in question_lower:
            patterns.add('min_query')
        
        # Relationship indicators
        if 'with their' in question_lower or 'and their' in question_lower:
            patterns.add('relationship_query')
        if 'by' in question_lower:
            patterns.add('grouping_query')
        
        return patterns
    
    def _extract_keywords_from_question(self, question: str) -> List[str]:
        """Extract keywords from question"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'when', 'where', 'who', 'why', 'how', 'which'
        }
        
        words = re.findall(r'\w+', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:8]  # Limit to 8 keywords
    
    def _classify_query_type(self, sql: str) -> str:
        """Classify the type of SQL query"""
        sql_upper = sql.upper()
        
        if 'COUNT(' in sql_upper:
            return 'count'
        elif 'AVG(' in sql_upper:
            return 'average'
        elif 'SUM(' in sql_upper:
            return 'sum'
        elif 'MAX(' in sql_upper or 'MIN(' in sql_upper:
            return 'minmax'
        elif 'JOIN' in sql_upper:
            return 'join'
        elif 'GROUP BY' in sql_upper:
            return 'group_by'
        elif 'ORDER BY' in sql_upper:
            return 'ordered_select'
        elif 'UNION' in sql_upper or 'INTERSECT' in sql_upper or 'EXCEPT' in sql_upper:
            return 'set_operation'
        else:
            return 'basic_select'
    
    def _extract_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """Extract common patterns from query history"""
        patterns = {
            "question_patterns": {},
            "sql_patterns": {},
            "database_usage": {},
            "success_rate": 0.0,
            "query_types": {},
            "performance_stats": {},
            "temporal_patterns": {}
        }
        
        if not self.query_history:
            return patterns
        
        # Analyze question patterns
        for query in self.query_history:
            question = query['question'].lower()
            
            # Question type patterns
            if question.startswith('how many'):
                patterns["question_patterns"]["count"] = patterns["question_patterns"].get("count", 0) + 1
            elif question.startswith('list') or question.startswith('show'):
                patterns["question_patterns"]["list"] = patterns["question_patterns"].get("list", 0) + 1
            elif 'average' in question or 'mean' in question:
                patterns["question_patterns"]["average"] = patterns["question_patterns"].get("average", 0) + 1
            elif 'with their' in question or 'and their' in question:
                patterns["question_patterns"]["join"] = patterns["question_patterns"].get("join", 0) + 1
            elif question.startswith('what'):
                patterns["question_patterns"]["what"] = patterns["question_patterns"].get("what", 0) + 1
        
        # Analyze SQL patterns
        for query in self.query_history:
            sql = query['sql'].upper()
            
            if 'COUNT(' in sql:
                patterns["sql_patterns"]["count"] = patterns["sql_patterns"].get("count", 0) + 1
            if 'JOIN' in sql:
                patterns["sql_patterns"]["join"] = patterns["sql_patterns"].get("join", 0) + 1
            if 'GROUP BY' in sql:
                patterns["sql_patterns"]["group_by"] = patterns["sql_patterns"].get("group_by", 0) + 1
            if 'ORDER BY' in sql:
                patterns["sql_patterns"]["order_by"] = patterns["sql_patterns"].get("order_by", 0) + 1
            if 'AVG(' in sql:
                patterns["sql_patterns"]["average"] = patterns["sql_patterns"].get("average", 0) + 1
        
        # Database usage
        for query in self.query_history:
            db = query['db_name']
            patterns["database_usage"][db] = patterns["database_usage"].get(db, 0) + 1
        
        # Query types
        for query in self.query_history:
            query_type = query.get('query_type', 'unknown')
            patterns["query_types"][query_type] = patterns["query_types"].get(query_type, 0) + 1
        
        # Success rate
        successful = sum(1 for q in self.query_history if q.get('success', True))
        patterns["success_rate"] = successful / len(self.query_history)
        
        # Performance statistics
        execution_times = [q.get('execution_time') for q in self.query_history if q.get('execution_time') is not None]
        if execution_times:
            patterns["performance_stats"] = {
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "total_queries_with_timing": len(execution_times)
            }
        
        return patterns
    
    def _get_performance_stats(self, db_name: str = None) -> Dict[str, Any]:
        """Get performance statistics for queries"""
        filtered_queries = self.query_history
        
        if db_name:
            filtered_queries = [q for q in self.query_history if q.get('db_name') == db_name]
        
        if not filtered_queries:
            return {"error": f"No queries found{' for database ' + db_name if db_name else ''}"}
        
        stats = {
            "total_queries": len(filtered_queries),
            "successful_queries": sum(1 for q in filtered_queries if q.get('success', True)),
            "database": db_name or "all"
        }
        
        # Execution time statistics
        execution_times = [q.get('execution_time') for q in filtered_queries if q.get('execution_time') is not None]
        if execution_times:
            stats["execution_time_stats"] = {
                "count": len(execution_times),
                "average": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "median": sorted(execution_times)[len(execution_times) // 2]
            }
        
        # Confidence statistics
        confidence_scores = [q.get('confidence') for q in filtered_queries if q.get('confidence') is not None]
        if confidence_scores:
            stats["confidence_stats"] = {
                "count": len(confidence_scores),
                "average": sum(confidence_scores) / len(confidence_scores),
                "min": min(confidence_scores),
                "max": max(confidence_scores)
            }
        
        # Query type distribution
        query_types = {}
        for query in filtered_queries:
            query_type = query.get('query_type', 'unknown')
            query_types[query_type] = query_types.get(query_type, 0) + 1
        
        stats["query_type_distribution"] = query_types
        stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
        
        return stats
    
    def _get_recent_queries(self, limit: int) -> List[Dict]:
        """Get recent queries from history"""
        # Sort by timestamp (most recent first)
        sorted_queries = sorted(
            self.query_history, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        
        return sorted_queries[:limit]

# ================================
# Server Factory and Runner
# ================================

def create_server(server_type: str):
    """Factory function to create MCP servers"""
    if server_type == "database_schema":
        return DatabaseSchemaServer()
    elif server_type == "sql_docs":
        return SQLDocumentationServer()
    elif server_type == "query_history":
        return QueryHistoryServer()
    else:
        raise ValueError(f"Unknown server type: {server_type}")

async def run_database_schema_server():
    """Run the database schema MCP server"""
    if not MCP_SERVER_AVAILABLE:
        print("❌ MCP Server not available")
        return
    
    server = DatabaseSchemaServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(read_stream, write_stream, server.server.create_initialization_options())

async def run_sql_docs_server():
    """Run the SQL documentation MCP server"""
    if not MCP_SERVER_AVAILABLE:
        print("❌ MCP Server not available")
        return
    
    server = SQLDocumentationServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(read_stream, write_stream, server.server.create_initialization_options())

async def run_query_history_server():
    """Run the query history MCP server"""
    if not MCP_SERVER_AVAILABLE:
        print("❌ MCP Server not available")
        return
    
    server = QueryHistoryServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(read_stream, write_stream, server.server.create_initialization_options())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python servers.py <server_type>")
        print("Server types: database_schema, sql_docs, query_history")
        sys.exit(1)
    
    server_type = sys.argv[1]
    
    if server_type == "database_schema":
        asyncio.run(run_database_schema_server())
    elif server_type == "sql_docs":
        asyncio.run(run_sql_docs_server())
    elif server_type == "query_history":
        asyncio.run(run_query_history_server())
    else:
        print(f"Unknown server type: {server_type}")
        print("Available types: database_schema, sql_docs, query_history")
        sys.exit(1)