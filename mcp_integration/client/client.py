#!/usr/bin/env python3
"""
MCP Client implementation for Text2SQL
Connects to MCP servers and provides enhanced SQL generation

"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

# MCP core imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import (
        CallToolRequest, 
        ListToolsRequest,
        Tool
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not available. Install with: pip install mcp")

@dataclass
class MCPConfig:
    """Configuration for MCP servers"""
    name: str
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}

class MCPManager:
    """
    Manager for Model Context Protocol connections
    Handles multiple MCP servers for different data sources
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: Dict[str, MCPConfig] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._setup_default_servers()
    
    def load_config(self, config_path: str):
        """Load MCP server configurations from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            for server_name, server_config in config.get('mcpServers', {}).items():
                self.servers[server_name] = MCPConfig(
                    name=server_name,
                    command=server_config.get('command', ''),
                    args=server_config.get('args', []),
                    env=server_config.get('env', {})
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load MCP config: {e}")
            self._setup_default_servers()
    
    def _setup_default_servers(self):
        """Setup default MCP servers for Text2SQL"""
        # Database Schema Server
        self.servers['database_schema'] = MCPConfig(
            name='database_schema',
            command='python',
            args=['-m', 'mcp.servers', 'database_schema'],
            env={'MCP_SERVER_TYPE': 'database_schema'}
        )
        
        # SQL Documentation Server
        self.servers['sql_docs'] = MCPConfig(
            name='sql_docs',
            command='python',
            args=['-m', 'mcp.servers', 'sql_docs'],
            env={'MCP_SERVER_TYPE': 'sql_docs'}
        )
        
        # Query History Server
        self.servers['query_history'] = MCPConfig(
            name='query_history',
            command='python',
            args=['-m', 'mcp.servers', 'query_history'],
            env={'MCP_SERVER_TYPE': 'query_history'}
        )
    
    async def connect_server(self, server_name: str) -> bool:
        """Connect to an MCP server"""
        if not MCP_AVAILABLE:
            self.logger.warning("MCP not available")
            return False
        
        if server_name not in self.servers:
            self.logger.error(f"Server {server_name} not configured")
            return False
        
        if server_name in self.sessions:
            self.logger.info(f"Server {server_name} already connected")
            return True
        
        try:
            server_config = self.servers[server_name]
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env
            )
            
            # Create and connect session
            session = await stdio_client(server_params)
            await session.initialize()
            
            self.sessions[server_name] = session
            self.logger.info(f"Connected to MCP server: {server_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {server_name}: {e}")
            return False
    
    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.sessions:
            try:
                await self.sessions[server_name].close()
                del self.sessions[server_name]
                self.logger.info(f"Disconnected from {server_name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from {server_name}: {e}")
    
    async def list_tools(self, server_name: str) -> List[Tool]:
        """List available tools from a server"""
        if server_name not in self.sessions:
            await self.connect_server(server_name)
        
        if server_name not in self.sessions:
            return []
        
        try:
            response = await self.sessions[server_name].list_tools(ListToolsRequest())
            return response.tools
        except Exception as e:
            self.logger.error(f"Failed to list tools from {server_name}: {e}")
            return []
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Call a tool on a specific server"""
        if server_name not in self.sessions:
            await self.connect_server(server_name)
        
        if server_name not in self.sessions:
            return None
        
        try:
            request = CallToolRequest(name=tool_name, arguments=arguments)
            response = await self.sessions[server_name].call_tool(request)
            
            # Extract content from response
            if hasattr(response, 'content') and response.content:
                # Get the text content from the first content item
                content = response.content[0]
                if hasattr(content, 'text'):
                    return json.loads(content.text)
                else:
                    return content
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            return None
    
    # High-level convenience methods
    async def get_database_schema(self, db_path: str) -> Optional[Dict[str, Any]]:
        """Get database schema using MCP"""
        return await self.call_tool(
            'database_schema',
            'get_schema',
            {'database_path': db_path}
        )
    
    async def get_table_relationships(self, db_path: str, table_name: str) -> Optional[Dict]:
        """Get table relationships and foreign keys"""
        return await self.call_tool(
            'database_schema',
            'get_relationships',
            {'database_path': db_path, 'table_name': table_name}
        )
    
    async def analyze_column_types(self, db_path: str, table_name: str) -> Optional[Dict]:
        """Analyze column types and constraints"""
        return await self.call_tool(
            'database_schema',
            'analyze_column_types',
            {'database_path': db_path, 'table_name': table_name}
        )
    
    async def get_sql_syntax_help(self, keyword: str) -> Optional[Dict]:
        """Get SQL syntax help for a keyword"""
        return await self.call_tool(
            'sql_docs',
            'get_syntax_help',
            {'sql_keyword': keyword}
        )
    
    async def validate_sql(self, sql_query: str) -> Optional[Dict]:
        """Validate SQL syntax and get suggestions"""
        return await self.call_tool(
            'sql_docs',
            'validate_sql',
            {'sql_query': sql_query}
        )
    
    async def get_best_practices(self, topic: str) -> Optional[Dict]:
        """Get SQL best practices for a topic"""
        return await self.call_tool(
            'sql_docs',
            'get_best_practices',
            {'topic': topic}
        )
    
    async def explain_sql(self, sql_query: str) -> Optional[Dict]:
        """Explain what a SQL query does"""
        return await self.call_tool(
            'sql_docs',
            'explain_sql',
            {'sql_query': sql_query}
        )
    
    async def search_query_history(self, keywords: List[str], limit: int = 10) -> Optional[List[Dict]]:
        """Search historical queries"""
        return await self.call_tool(
            'query_history',
            'search_queries',
            {'keywords': keywords, 'limit': limit}
        )
    
    async def add_query_to_history(self, question: str, sql: str, db_name: str, success: bool = True) -> Optional[Dict]:
        """Add a query to history"""
        return await self.call_tool(
            'query_history',
            'add_query',
            {'question': question, 'sql': sql, 'db_name': db_name, 'success': success}
        )
    
    async def get_similar_queries(self, question: str, limit: int = 5) -> Optional[List[Dict]]:
        """Find similar queries based on question patterns"""
        return await self.call_tool(
            'query_history',
            'get_similar_queries',
            {'question': question, 'limit': limit}
        )
    
    async def get_query_patterns(self, pattern_type: str = "all") -> Optional[Dict]:
        """Get common query patterns from history"""
        return await self.call_tool(
            'query_history',
            'get_query_patterns',
            {'pattern_type': pattern_type}
        )
    
    async def close_all(self):
        """Close all MCP connections"""
        for server_name in list(self.sessions.keys()):
            await self.disconnect_server(server_name)

class MCPEnhancedSQLGenerator:
    """
    SQL Generator enhanced with MCP capabilities
    Uses MCP to gather context and improve SQL generation
    """
    
    def __init__(self, mcp_manager: MCPManager, base_generator=None):
        self.mcp_manager = mcp_manager
        self.base_generator = base_generator
        self.logger = logging.getLogger(__name__)
    
    async def generate_sql_with_context(self, 
                                       question: str, 
                                       db_path: str,
                                       use_mcp: bool = True) -> Dict[str, Any]:
        """
        Generate SQL with enhanced context from MCP
        
        Returns:
            Dict containing:
            - sql: Generated SQL query
            - context: Context used for generation
            - confidence: Confidence score
            - sources: MCP sources used
        """
        result = {
            'sql': '',
            'context': {},
            'confidence': 0.0,
            'sources': [],
            'mcp_enhanced': use_mcp
        }
        
        if not use_mcp or not MCP_AVAILABLE:
            # Fallback to base generator
            if self.base_generator:
                result['sql'] = self.base_generator.generate_sql_from_question(question, db_path)
                result['confidence'] = 0.5
            else:
                result['sql'] = self._pattern_generate_sql(question, {})
                result['confidence'] = 0.3
            return result
        
        try:
            # Step 1: Get enhanced database schema
            schema_context = await self.mcp_manager.get_database_schema(db_path)
            if schema_context:
                result['context']['schema'] = schema_context
                result['sources'].append('database_schema')
            
            # Step 2: Analyze question to determine query type and get syntax help
            query_keywords = self._extract_sql_keywords(question)
            for keyword in query_keywords:
                syntax_help = await self.mcp_manager.get_sql_syntax_help(keyword)
                if syntax_help:
                    if 'syntax_help' not in result['context']:
                        result['context']['syntax_help'] = {}
                    result['context']['syntax_help'][keyword] = syntax_help
                    if 'sql_docs' not in result['sources']:
                        result['sources'].append('sql_docs')
            
            # Step 3: Search for similar historical queries
            keywords = self._extract_keywords(question)
            history = await self.mcp_manager.search_query_history(keywords)
            if history:
                result['context']['history'] = history
                result['sources'].append('query_history')
            
            # Step 4: Generate SQL with enhanced context
            result['sql'] = await self._generate_enhanced_sql(
                question, db_path, result['context']
            )
            
            # Step 5: Validate generated SQL
            if result['sql']:
                validation = await self.mcp_manager.validate_sql(result['sql'])
                if validation:
                    result['context']['validation'] = validation
                    
                    # Adjust confidence based on validation
                    if validation.get('is_valid', False):
                        result['confidence'] += 0.2
                    else:
                        result['confidence'] -= 0.1
            
            # Step 6: Calculate confidence based on available context
            result['confidence'] = self._calculate_confidence(result)
            
        except Exception as e:
            self.logger.error(f"MCP-enhanced generation failed: {e}")
            # Fallback to base generator
            if self.base_generator:
                result['sql'] = self.base_generator.generate_sql_from_question(question, db_path)
                result['confidence'] = 0.3
            else:
                result['sql'] = self._pattern_generate_sql(question, result['context'].get('schema', {}))
                result['confidence'] = 0.2
        
        return result
    
    def _extract_sql_keywords(self, question: str) -> List[str]:
        """Extract SQL-related keywords from question"""
        question_lower = question.lower()
        keywords = []
        
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            keywords.append('select')
        elif any(word in question_lower for word in ['list', 'show', 'display', 'get']):
            keywords.append('select')
        elif any(word in question_lower for word in ['with their', 'and their', 'join']):
            keywords.append('join')
        elif any(word in question_lower for word in ['group by', 'grouped', 'by category']):
            keywords.append('group by')
        elif any(word in question_lower for word in ['order by', 'sorted', 'ordered']):
            keywords.append('order by')
        elif 'where' in question_lower:
            keywords.append('where')
        
        return keywords
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from question for history search"""
        # Simple keyword extraction - can be enhanced with NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = question.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:5]  # Limit to top 5 keywords
    
    async def _generate_enhanced_sql(self, 
                                   question: str, 
                                   db_path: str, 
                                   context: Dict[str, Any]) -> str:
        """Generate SQL using enhanced context from MCP"""
        
        # Use schema context if available
        schema_info = context.get('schema', {})
        
        # Use history context for pattern matching
        history = context.get('history', [])
        
        # Try to find exact match in history first
        if history:
            for hist_query in history[:3]:  # Check top 3 most relevant
                if hist_query.get('relevance_score', 0) > 2.0:  # High relevance
                    # Use similar SQL as base, but adapt to current question
                    base_sql = hist_query.get('sql', '')
                    adapted_sql = self._adapt_sql_from_history(base_sql, question, schema_info)
                    if adapted_sql:
                        return adapted_sql
        
        # Generate new SQL using schema and patterns
        return self._generate_sql_from_schema(question, schema_info)
    
    def _adapt_sql_from_history(self, base_sql: str, question: str, schema: Dict) -> str:
        """Adapt historical SQL to current question"""
        # This is a simplified adaptation - can be enhanced
        # For now, just return the base SQL if it seems appropriate
        question_lower = question.lower()
        base_sql_lower = base_sql.lower()
        
        # Check if the SQL type matches the question
        if 'count' in question_lower and 'count(' in base_sql_lower:
            return base_sql
        elif any(word in question_lower for word in ['list', 'show']) and 'select' in base_sql_lower:
            return base_sql
        elif 'join' in base_sql_lower and any(word in question_lower for word in ['with', 'and their']):
            return base_sql
        
        return None
    
    def _generate_sql_from_schema(self, question: str, schema: Dict) -> str:
        """Generate SQL using schema information"""
        question_lower = question.lower()
        
        if not schema or 'tables' not in schema:
            return self._pattern_generate_sql(question, {})
        
        tables = schema['tables']
        table_names = list(tables.keys())
        
        if not table_names:
            return "SELECT 1"
        
        # Use existing base_generator's table matching if available
        if self.base_generator and hasattr(self.base_generator, '_find_best_table_match'):
            primary_table = self.base_generator._find_best_table_match(question, schema)
        else:
            # Simple fallback - just use first table
            primary_table = table_names[0]
        
        # Generate SQL based on question type
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            return f"SELECT COUNT(*) FROM {primary_table}"
        
        elif any(word in question_lower for word in ['list', 'show', 'display']):
            # Check if question asks for specific columns
            table_columns = tables.get(primary_table, {}).get('columns', {})
            
            if 'name' in question_lower and any('name' in col.lower() for col in table_columns):
                name_col = next((col for col in table_columns if 'name' in col.lower()), '*')
                return f"SELECT {name_col} FROM {primary_table}"
            else:
                return f"SELECT * FROM {primary_table}"
        
        elif any(word in question_lower for word in ['with their', 'and their']):
            # This suggests a JOIN - find related tables
            join_sql = self._generate_join_sql(question, primary_table, tables, schema)
            if join_sql:
                return join_sql
        
        elif any(word in question_lower for word in ['average', 'mean']):
            # Find numeric columns
            table_columns = tables.get(primary_table, {}).get('columns', {})
            numeric_col = self._find_numeric_column(table_columns)
            if numeric_col:
                return f"SELECT AVG({numeric_col}) FROM {primary_table}"
        
        # Default fallback
        return f"SELECT * FROM {primary_table}"
    

    
    def _generate_join_sql(self, question: str, primary_table: str, tables: Dict, schema: Dict) -> Optional[str]:
        """Generate JOIN SQL based on relationships"""
        relationships = schema.get('relationships', {})
        
        # Look for foreign key relationships
        primary_table_info = tables.get(primary_table, {})
        foreign_keys = primary_table_info.get('foreign_keys', [])
        
        if foreign_keys:
            # Simple JOIN with first related table
            fk = foreign_keys[0]
            related_table = fk.get('references_table')
            if related_table and related_table in tables:
                from_col = fk.get('column')
                to_col = fk.get('references_column')
                
                return f"SELECT * FROM {primary_table} p JOIN {related_table} r ON p.{from_col} = r.{to_col}"
        
        return None
    
    def _find_numeric_column(self, columns: Dict) -> Optional[str]:
        """Find a numeric column in the table"""
        numeric_types = ['INTEGER', 'REAL', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE']
        
        for col_name, col_info in columns.items():
            col_type = col_info.get('type', '').upper()
            if any(num_type in col_type for num_type in numeric_types):
                return col_name
        
        # Look for common numeric column names
        numeric_names = ['age', 'salary', 'price', 'amount', 'count', 'value', 'score']
        for col_name in columns:
            if any(name in col_name.lower() for name in numeric_names):
                return col_name
        
        return None
    
    def _pattern_generate_sql(self, question: str, schema: Dict) -> str:
        """Generate SQL using simple patterns as fallback"""
        question_lower = question.lower()
        
        # Use existing base_generator if available
        if self.base_generator and hasattr(self.base_generator, '_pattern_generate_sql'):
            return self.base_generator._pattern_generate_sql(question, schema)
        
        # Simple fallback implementation
        if schema and 'tables' in schema:
            table_names = list(schema['tables'].keys())
            primary_table = table_names[0] if table_names else 'table_name'
        else:
            primary_table = 'table_name'
        
        if any(word in question_lower for word in ['how many', 'count']):
            return f"SELECT COUNT(*) FROM {primary_table}"
        elif any(word in question_lower for word in ['list', 'show']):
            return f"SELECT * FROM {primary_table}"
        elif 'name' in question_lower:
            return f"SELECT name FROM {primary_table}"
        else:
            return f"SELECT * FROM {primary_table}"
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on available context"""
        base_confidence = 0.3
        context_bonus = 0.0
        
        if result['context'].get('schema'):
            context_bonus += 0.3
        if result['context'].get('syntax_help'):
            context_bonus += 0.1
        if result['context'].get('history'):
            context_bonus += 0.2
        if result['context'].get('validation', {}).get('is_valid'):
            context_bonus += 0.2
        
        return min(1.0, base_confidence + context_bonus)

class MCPEnhancedEvaluator:
    """
    Enhanced evaluator that uses MCP for improved SQL generation
    """
    
    def __init__(self, base_evaluator, mcp_config_path: Optional[str] = None):
        self.base_evaluator = base_evaluator
        self.mcp_manager = MCPManager(mcp_config_path)
        self.sql_generator = MCPEnhancedSQLGenerator(self.mcp_manager, base_evaluator)
        self.logger = logging.getLogger(__name__)
    
    async def generate_sql_from_question(self, question: str, db_path: str, use_mcp: bool = True) -> str:
        """Generate SQL using MCP enhancement"""
        result = await self.sql_generator.generate_sql_with_context(question, db_path, use_mcp)
        
        # Log MCP usage
        if result['mcp_enhanced']:
            self.logger.info(f"MCP sources used: {result['sources']}")
            self.logger.info(f"Confidence: {result['confidence']:.2f}")
        
        # Add query to history for future learning
        if result['sql'] and use_mcp:
            try:
                db_name = Path(db_path).stem
                await self.mcp_manager.add_query_to_history(
                    question=question,
                    sql=result['sql'],
                    db_name=db_name,
                    success=True  # Assume success for now
                )
            except Exception as e:
                self.logger.warning(f"Failed to add query to history: {e}")
        
        return result['sql']
    
    async def evaluate_with_mcp(self, 
                               questions_file: str, 
                               gold_file: str, 
                               db_dir: str, 
                               **kwargs) -> Dict[str, Any]:
        """
        Evaluate SQL generation with MCP enhancement
        """
        try:
            # Connect to MCP servers
            await self.mcp_manager.connect_server('database_schema')
            await self.mcp_manager.connect_server('sql_docs')
            await self.mcp_manager.connect_server('query_history')
            
            # Load questions
            with open(questions_file, 'r') as f:
                question_lines = f.readlines()
            
            generated_predictions = []
            mcp_stats = {
                'total_queries': 0,
                'mcp_enhanced': 0,
                'confidence_scores': [],
                'source_usage': {},
                'validation_results': {'valid': 0, 'invalid': 0}
            }
            
            for line_num, line in enumerate(question_lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse question and database
                parts = line.split('\t') if '\t' in line else line.rsplit(' ', 1)
                if len(parts) < 2:
                    continue
                
                question = parts[0].strip()
                db_name = parts[1].strip()
                db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
                
                # Generate SQL with MCP enhancement
                result = await self.sql_generator.generate_sql_with_context(question, db_path)
                generated_predictions.append([result['sql']])
                
                # Track statistics
                mcp_stats['total_queries'] += 1
                if result['mcp_enhanced']:
                    mcp_stats['mcp_enhanced'] += 1
                
                mcp_stats['confidence_scores'].append(result['confidence'])
                
                # Track source usage
                for source in result['sources']:
                    mcp_stats['source_usage'][source] = mcp_stats['source_usage'].get(source, 0) + 1
                
                # Track validation results
                validation = result['context'].get('validation', {})
                if validation.get('is_valid'):
                    mcp_stats['validation_results']['valid'] += 1
                else:
                    mcp_stats['validation_results']['invalid'] += 1
                
                print(f"Q{line_num}: {question[:50]}...")
                print(f"SQL: {result['sql']}")
                print(f"Sources: {result['sources']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print()
            
            # Calculate MCP statistics
            if mcp_stats['confidence_scores']:
                avg_confidence = sum(mcp_stats['confidence_scores']) / len(mcp_stats['confidence_scores'])
                mcp_enhancement_rate = mcp_stats['mcp_enhanced'] / mcp_stats['total_queries']
                
                print(f"\n🤖 MCP Enhancement Statistics:")
                print(f"  Total queries: {mcp_stats['total_queries']}")
                print(f"  MCP enhanced: {mcp_stats['mcp_enhanced']} ({mcp_enhancement_rate:.1%})")
                print(f"  Average confidence: {avg_confidence:.2f}")
                print(f"  Source usage: {mcp_stats['source_usage']}")
                print(f"  Valid SQL: {mcp_stats['validation_results']['valid']}")
                print(f"  Invalid SQL: {mcp_stats['validation_results']['invalid']}")
            
            return {
                'predictions': generated_predictions,
                'mcp_stats': mcp_stats
            }
            
        finally:
            # Clean up MCP connections
            await self.mcp_manager.close_all()

# Async context manager for MCP
class MCPContext:
    """Async context manager for MCP operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.manager = MCPManager(config_path)
    
    async def __aenter__(self):
        # Connect to default servers
        await self.manager.connect_server('database_schema')
        await self.manager.connect_server('sql_docs')
        await self.manager.connect_server('query_history')
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.close_all()

# Utility functions
async def quick_mcp_test():
    """Quick test of MCP functionality"""
    async with MCPContext() as mcp:
        # Test database schema retrieval
        schema = await mcp.get_database_schema("test.db")
        print(f"Schema: {schema}")
        
        # Test SQL syntax help
        help_info = await mcp.get_sql_syntax_help("select")
        print(f"Help: {help_info}")

if __name__ == "__main__":
    # Example usage
    asyncio.run(quick_mcp_test())