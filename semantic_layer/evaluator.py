"""
Semantic Enhanced Evaluator Module
==================================

Enhanced evaluator that extends the existing evaluation system with semantic understanding
capabilities for improved Text-to-SQL generation.

Classes:
    - SemanticEvaluator: Enhanced evaluator with semantic layer integration

Functions:
    - evaluate_with_semantics: Convenience function for semantic evaluation
    - batch_evaluate_with_semantics: Batch evaluation with semantics
"""

import os
import sys
import sqlite3
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Import the semantic layer core
try:
    from .core import SimpleSemanticLayer, create_semantic_layer, enhance_sql_generation
    SEMANTIC_CORE_AVAILABLE = True
except ImportError:
    try:
        from core import SimpleSemanticLayer, create_semantic_layer, enhance_sql_generation
        SEMANTIC_CORE_AVAILABLE = True
    except ImportError:
        print("Warning: Semantic layer core not available")
        SEMANTIC_CORE_AVAILABLE = False
        SimpleSemanticLayer = None

# Try to import existing evaluator
try:
    from utils.eval import Evaluator
    EXISTING_EVAL_AVAILABLE = True
except ImportError:
    print("Info: utils.eval not available. Using standalone mode.")
    EXISTING_EVAL_AVAILABLE = False
    # Create a dummy base class
    class Evaluator:
        def __init__(self, *args, **kwargs):
            pass

class SemanticEvaluator(Evaluator):
    """
    Enhanced evaluator that adds semantic understanding to SQL generation.
    Extends the existing Evaluator class with semantic layer capabilities.
    """
    
    def __init__(self, prompt_type="enhanced", enable_debugging=False, 
                 use_chromadb=False, chromadb_config=None, semantic_config_path=None):
        
        # Initialize parent class if available
        if EXISTING_EVAL_AVAILABLE:
            try:
                super().__init__(prompt_type, enable_debugging, use_chromadb, chromadb_config)
            except Exception as e:
                print(f"Warning: Could not initialize parent evaluator: {e}")
        
        # Initialize semantic layer
        if SEMANTIC_CORE_AVAILABLE:
            try:
                if semantic_config_path and os.path.exists(semantic_config_path):
                    self.semantic_layer = SimpleSemanticLayer(semantic_config_path)
                else:
                    self.semantic_layer = create_semantic_layer()
                self.semantic_enabled = True
                print("Semantic layer initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize semantic layer: {e}")
                self.semantic_layer = None
                self.semantic_enabled = False
        else:
            self.semantic_layer = None
            self.semantic_enabled = False
            print("Warning: Semantic layer not available")
        
        # Statistics tracking
        self.semantic_stats = {
            'queries_analyzed': 0,
            'queries_enhanced': 0,
            'suggestions_made': 0,
            'complexity_scores': [],
            'enhancement_types': {
                'count_enhancements': 0,
                'aggregation_enhancements': 0,
                'grouping_enhancements': 0,
                'join_enhancements': 0,
                'filtering_enhancements': 0,
                'ordering_enhancements': 0
            },
            'entity_detections': {
                'car_entities': 0,
                'student_entities': 0,
                'customer_entities': 0,
                'product_entities': 0,
                'other_entities': 0
            }
        }
        
        # Enhancement configuration
        self.enhancement_config = {
            'enable_count_enhancement': True,
            'enable_aggregation_enhancement': True,
            'enable_grouping_enhancement': True,
            'enable_join_suggestions': True,
            'enable_filtering_enhancement': True,
            'enable_ordering_enhancement': True,
            'max_suggestions_per_query': 5
        }
    
    def generate_sql_from_question(self, question: str, db_path: str) -> str:
        """
        Generate SQL with semantic enhancement.
        Overrides the parent method to add semantic understanding.
        """
        # Generate base SQL using parent method if available
        base_sql = ""
        if EXISTING_EVAL_AVAILABLE and hasattr(super(), 'generate_sql_from_question'):
            try:
                base_sql = super().generate_sql_from_question(question, db_path)
            except Exception as e:
                print(f"Warning: Parent SQL generation failed: {e}")
                base_sql = self._simple_sql_generation(question, db_path)
        else:
            base_sql = self._simple_sql_generation(question, db_path)
        
        # Apply semantic enhancement
        if self.semantic_enabled:
            enhanced_sql = self._apply_semantic_enhancement(question, base_sql, db_path)
            
            # Update statistics
            self.semantic_stats['queries_analyzed'] += 1
            if enhanced_sql != base_sql:
                self.semantic_stats['queries_enhanced'] += 1
            
            return enhanced_sql
        
        return base_sql
    
    def generate_enhanced_sql(self, question: str, db_path: str) -> Dict[str, Any]:
        """
        Generate SQL with comprehensive semantic enhancement and analysis.
        Returns detailed information about the enhancement process.
        """
        result = {
            'question': question,
            'base_sql': '',
            'enhanced_sql': '',
            'semantic_analysis': None,
            'suggestions': [],
            'enhancement_applied': False,
            'complexity': 'unknown',
            'recommended_approach': 'unknown',
            'enhancement_types': [],
            'confidence_score': 0.0
        }
        
        # Generate base SQL
        try:
            result['base_sql'] = self._generate_base_sql(question, db_path)
        except Exception as e:
            print(f"Warning: Base SQL generation failed: {e}")
            result['base_sql'] = "SELECT 1"
        
        # Apply semantic enhancement if enabled
        if self.semantic_enabled:
            try:
                enhancement_result = self._comprehensive_semantic_enhancement(
                    question, result['base_sql'], db_path
                )
                result.update(enhancement_result)
            except Exception as e:
                print(f"Warning: Semantic enhancement failed: {e}")
                result['enhanced_sql'] = result['base_sql']
        else:
            result['enhanced_sql'] = result['base_sql']
        
        return result
    
    def _generate_base_sql(self, question: str, db_path: str) -> str:
        """Generate base SQL using available methods"""
        
        # Try parent method first
        if EXISTING_EVAL_AVAILABLE and hasattr(super(), 'generate_sql_from_question'):
            try:
                return super().generate_sql_from_question(question, db_path)
            except Exception as e:
                print(f"Parent SQL generation failed: {e}")
        
        # Fallback to simple generation
        return self._simple_sql_generation(question, db_path)
    
    def _comprehensive_semantic_enhancement(self, question: str, base_sql: str, db_path: str) -> Dict[str, Any]:
        """Apply comprehensive semantic enhancement with detailed analysis"""
        
        # Get database schema
        schema_info = self._get_database_schema(db_path)
        
        # Get semantic analysis
        semantic_context = self.semantic_layer.get_semantic_context(question, schema_info)
        semantic_analysis = semantic_context['semantic_analysis']
        
        # Apply various enhancement techniques
        enhanced_sql = base_sql
        enhancement_types = []
        suggestions = []
        
        # Count enhancement
        if self.enhancement_config['enable_count_enhancement']:
            count_result = self._apply_count_enhancement(enhanced_sql, question, semantic_analysis, schema_info)
            if count_result['enhanced']:
                enhanced_sql = count_result['sql']
                enhancement_types.append('count')
                suggestions.extend(count_result['suggestions'])
                self.semantic_stats['enhancement_types']['count_enhancements'] += 1
        
        # Aggregation enhancement
        if self.enhancement_config['enable_aggregation_enhancement']:
            agg_result = self._apply_aggregation_enhancement(enhanced_sql, question, semantic_analysis, schema_info)
            if agg_result['enhanced']:
                enhanced_sql = agg_result['sql']
                enhancement_types.append('aggregation')
                suggestions.extend(agg_result['suggestions'])
                self.semantic_stats['enhancement_types']['aggregation_enhancements'] += 1
        
        # Grouping enhancement
        if self.enhancement_config['enable_grouping_enhancement']:
            group_result = self._apply_grouping_enhancement(enhanced_sql, question, semantic_analysis, schema_info)
            if group_result['enhanced']:
                enhanced_sql = group_result['sql']
                enhancement_types.append('grouping')
                suggestions.extend(group_result['suggestions'])
                self.semantic_stats['enhancement_types']['grouping_enhancements'] += 1
        
        # Join enhancement
        if self.enhancement_config['enable_join_suggestions']:
            join_result = self._apply_join_enhancement(enhanced_sql, question, semantic_analysis, schema_info)
            if join_result['enhanced']:
                enhanced_sql = join_result['sql']
                enhancement_types.append('join')
                suggestions.extend(join_result['suggestions'])
                self.semantic_stats['enhancement_types']['join_enhancements'] += 1
        
        # Filtering enhancement
        if self.enhancement_config['enable_filtering_enhancement']:
            filter_result = self._apply_filtering_enhancement(enhanced_sql, question, semantic_analysis, schema_info)
            if filter_result['enhanced']:
                enhanced_sql = filter_result['sql']
                enhancement_types.append('filtering')
                suggestions.extend(filter_result['suggestions'])
                self.semantic_stats['enhancement_types']['filtering_enhancements'] += 1
        
        # Ordering enhancement
        if self.enhancement_config['enable_ordering_enhancement']:
            order_result = self._apply_ordering_enhancement(enhanced_sql, question, semantic_analysis, schema_info)
            if order_result['enhanced']:
                enhanced_sql = order_result['sql']
                enhancement_types.append('ordering')
                suggestions.extend(order_result['suggestions'])
                self.semantic_stats['enhancement_types']['ordering_enhancements'] += 1
        
        # Track entity detections
        self._track_entity_detections(semantic_analysis)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(semantic_analysis, enhancement_types)
        
        # Update statistics
        self.semantic_stats['suggestions_made'] += len(suggestions)
        self.semantic_stats['complexity_scores'].append(semantic_analysis['complexity_score'])
        
        return {
            'enhanced_sql': enhanced_sql,
            'semantic_analysis': semantic_context,
            'suggestions': suggestions[:self.enhancement_config['max_suggestions_per_query']],
            'enhancement_applied': len(enhancement_types) > 0,
            'complexity': semantic_context['complexity'],
            'recommended_approach': semantic_context['recommended_approach'],
            'enhancement_types': enhancement_types,
            'confidence_score': confidence_score
        }
    
    def _apply_count_enhancement(self, sql: str, question: str, analysis: Dict, schema_info: Dict) -> Dict:
        """Apply count-specific enhancements"""
        result = {'enhanced': False, 'sql': sql, 'suggestions': []}
        
        # Check if count is needed
        count_metrics = [m for m in analysis['relevant_metrics'] if m['type'] == 'count']
        if not count_metrics:
            return result
        
        question_lower = question.lower()
        
        # Replace SELECT * with COUNT(*) if appropriate
        if 'SELECT *' in sql.upper() and any(pattern in question_lower for pattern in ['how many', 'count', 'number of']):
            enhanced_sql = sql.replace('SELECT *', 'SELECT COUNT(*)', 1)
            result['enhanced'] = True
            result['sql'] = enhanced_sql
            result['suggestions'].append("Replaced SELECT * with COUNT(*) for counting query")
        
        return result
    
    def _apply_aggregation_enhancement(self, sql: str, question: str, analysis: Dict, schema_info: Dict) -> Dict:
        """Apply aggregation-specific enhancements"""
        result = {'enhanced': False, 'sql': sql, 'suggestions': []}
        
        agg_metrics = [m for m in analysis['relevant_metrics'] if m['type'] in ['average', 'sum', 'max', 'min']]
        if not agg_metrics:
            return result
        
        question_lower = question.lower()
        
        for metric in agg_metrics:
            metric_type = metric['type']
            
            if 'SELECT *' in sql.upper():
                # Find relevant numeric columns
                relevant_columns = self._find_relevant_columns(question, schema_info, 'numeric')
                
                if relevant_columns:
                    col = relevant_columns[0]
                    
                    if metric_type == 'average' and any(word in question_lower for word in ['average', 'mean', 'avg']):
                        enhanced_sql = sql.replace('SELECT *', f'SELECT AVG({col})', 1)
                        result['enhanced'] = True
                        result['sql'] = enhanced_sql
                        result['suggestions'].append(f"Added AVG aggregation for {col}")
                        break
                    
                    elif metric_type == 'sum' and any(word in question_lower for word in ['total', 'sum']):
                        enhanced_sql = sql.replace('SELECT *', f'SELECT SUM({col})', 1)
                        result['enhanced'] = True
                        result['sql'] = enhanced_sql
                        result['suggestions'].append(f"Added SUM aggregation for {col}")
                        break
                    
                    elif metric_type == 'max' and any(word in question_lower for word in ['maximum', 'highest', 'top']):
                        enhanced_sql = sql.replace('SELECT *', f'SELECT MAX({col})', 1)
                        result['enhanced'] = True
                        result['sql'] = enhanced_sql
                        result['suggestions'].append(f"Added MAX aggregation for {col}")
                        break
                    
                    elif metric_type == 'min' and any(word in question_lower for word in ['minimum', 'lowest', 'bottom']):
                        enhanced_sql = sql.replace('SELECT *', f'SELECT MIN({col})', 1)
                        result['enhanced'] = True
                        result['sql'] = enhanced_sql
                        result['suggestions'].append(f"Added MIN aggregation for {col}")
                        break
        
        return result
    
    def _apply_grouping_enhancement(self, sql: str, question: str, analysis: Dict, schema_info: Dict) -> Dict:
        """Apply grouping-specific enhancements"""
        result = {'enhanced': False, 'sql': sql, 'suggestions': []}
        
        # Check if grouping is needed
        dimensions = analysis['relevant_dimensions']
        if not dimensions or 'GROUP BY' in sql.upper():
            return result
        
        question_lower = question.lower()
        
        # Check for grouping keywords
        if any(keyword in question_lower for keyword in ['by', 'per', 'group', 'each', 'every']):
            # Find relevant categorical columns
            categorical_cols = self._find_relevant_columns(question, schema_info, 'categorical')
            
            if categorical_cols:
                col = categorical_cols[0]
                enhanced_sql = sql.rstrip(';') + f' GROUP BY {col}'
                result['enhanced'] = True
                result['sql'] = enhanced_sql
                result['suggestions'].append(f"Added GROUP BY {col} for dimensional analysis")
            
            # Also try temporal grouping
            elif any(word in question_lower for word in ['year', 'month', 'day', 'time']):
                temporal_cols = self._find_relevant_columns(question, schema_info, 'temporal')
                if temporal_cols:
                    col = temporal_cols[0]
                    enhanced_sql = sql.rstrip(';') + f' GROUP BY {col}'
                    result['enhanced'] = True
                    result['sql'] = enhanced_sql
                    result['suggestions'].append(f"Added GROUP BY {col} for temporal analysis")
        
        return result
    
    def _apply_join_enhancement(self, sql: str, question: str, analysis: Dict, schema_info: Dict) -> Dict:
        """Apply join-specific enhancements"""
        result = {'enhanced': False, 'sql': sql, 'suggestions': []}
        
        entities = analysis['relevant_entities']
        if len(entities) <= 1:
            return result
        
        # Suggest joins based on entities
        suggested_joins = []
        
        for entity in entities:
            if 'relationships' in entity and entity['relationships']:
                for relationship in entity['relationships']:
                    join_condition = relationship.get('join', '')
                    if join_condition and join_condition not in sql:
                        suggested_joins.append(join_condition)
        
        if suggested_joins:
            # Add the first suggested join
            join = suggested_joins[0]
            table_match = re.search(r'(\w+)\.', join)
            if table_match:
                join_table = table_match.group(1)
                enhanced_sql = sql.rstrip(';') + f' LEFT JOIN {join_table} ON {join}'
                result['enhanced'] = True
                result['sql'] = enhanced_sql
                result['suggestions'].append(f"Added JOIN: {join}")
        
        return result
    
    def _apply_filtering_enhancement(self, sql: str, question: str, analysis: Dict, schema_info: Dict) -> Dict:
        """Apply filtering-specific enhancements"""
        result = {'enhanced': False, 'sql': sql, 'suggestions': []}
        
        question_lower = question.lower()
        
        # Check for comparison operators
        comparisons = {
            'greater than': '>',
            'more than': '>',
            'above': '>',
            'over': '>',
            'less than': '<',
            'under': '<',
            'below': '<',
            'equal to': '=',
            'equals': '=',
            'is': '='
        }
        
        for phrase, operator in comparisons.items():
            if phrase in question_lower and 'WHERE' not in sql.upper():
                # Try to extract value
                parts = question_lower.split(phrase)
                if len(parts) > 1:
                    # Simple value extraction
                    value_part = parts[1].strip().split()[0]
                    if value_part.replace('.', '').isdigit():
                        # Find relevant numeric column
                        numeric_cols = self._find_relevant_columns(question, schema_info, 'numeric')
                        if numeric_cols:
                            col = numeric_cols[0]
                            enhanced_sql = sql.rstrip(';') + f' WHERE {col} {operator} {value_part}'
                            result['enhanced'] = True
                            result['sql'] = enhanced_sql
                            result['suggestions'].append(f"Added WHERE condition: {col} {operator} {value_part}")
                            break
        
        return result
    
    def _apply_ordering_enhancement(self, sql: str, question: str, analysis: Dict, schema_info: Dict) -> Dict:
        """Apply ordering-specific enhancements"""
        result = {'enhanced': False, 'sql': sql, 'suggestions': []}
        
        question_lower = question.lower()
        
        if 'ORDER BY' in sql.upper():
            return result
        
        # Check for ordering keywords
        if any(word in question_lower for word in ['top', 'highest', 'maximum', 'largest']):
            # Find relevant numeric columns
            numeric_cols = self._find_relevant_columns(question, schema_info, 'numeric')
            if numeric_cols:
                col = numeric_cols[0]
                enhanced_sql = sql.rstrip(';') + f' ORDER BY {col} DESC'
                result['enhanced'] = True
                result['sql'] = enhanced_sql
                result['suggestions'].append(f"Added ORDER BY {col} DESC for top results")
        
        elif any(word in question_lower for word in ['bottom', 'lowest', 'minimum', 'smallest']):
            # Find relevant numeric columns
            numeric_cols = self._find_relevant_columns(question, schema_info, 'numeric')
            if numeric_cols:
                col = numeric_cols[0]
                enhanced_sql = sql.rstrip(';') + f' ORDER BY {col} ASC'
                result['enhanced'] = True
                result['sql'] = enhanced_sql
                result['suggestions'].append(f"Added ORDER BY {col} ASC for bottom results")
        
        # Add LIMIT for top/bottom queries
        if result['enhanced'] and any(word in question_lower for word in ['top', 'bottom']):
            # Try to extract number
            numbers = re.findall(r'\b(\d+)\b', question)
            if numbers:
                limit = numbers[0]
                result['sql'] += f' LIMIT {limit}'
                result['suggestions'].append(f"Added LIMIT {limit}")
        
        return result
    
    def _apply_semantic_enhancement(self, question: str, base_sql: str, db_path: str) -> str:
        """Apply semantic enhancement (simplified version for compatibility)"""
        if not self.semantic_enabled:
            return base_sql
        
        try:
            result = self._comprehensive_semantic_enhancement(question, base_sql, db_path)
            return result['enhanced_sql']
        except Exception as e:
            print(f"Warning: Semantic enhancement failed: {e}")
            return base_sql
    
    def _find_relevant_columns(self, question: str, schema_info: Dict, column_type: str) -> List[str]:
        """Find relevant columns based on question and type"""
        if not schema_info:
            return []
        
        question_lower = question.lower()
        relevant_columns = []
        
        # Column type indicators from config or defaults
        type_indicators = {
            'numeric': ['price', 'cost', 'amount', 'value', 'count', 'number', 'age', 'year',
                       'horsepower', 'mpg', 'weight', 'cylinders', 'grade', 'score', 'id'],
            'categorical': ['type', 'category', 'class', 'status', 'region', 'make', 'model',
                           'brand', 'name', 'department', 'level'],
            'temporal': ['date', 'time', 'year', 'month', 'day', 'created', 'updated'],
            'textual': ['name', 'description', 'comment', 'text', 'address', 'email']
        }
        
        indicators = type_indicators.get(column_type, [])
        
        for table, columns in schema_info.items():
            for column in columns:
                column_lower = column.lower()
                
                # Check if column matches question terms
                question_words = question_lower.split()
                if any(word in column_lower for word in question_words):
                    # Check if it matches the requested type
                    if any(indicator in column_lower for indicator in indicators):
                        relevant_columns.append(f"{table}.{column}")
                
                # Also check direct type match
                elif any(indicator in column_lower for indicator in indicators):
                    relevant_columns.append(f"{table}.{column}")
        
        return relevant_columns
    
    def _track_entity_detections(self, analysis: Dict):
        """Track entity detections for statistics"""
        entities = analysis['relevant_entities']
        
        for entity in entities:
            entity_name = entity['name'].lower()
            if 'car' in entity_name:
                self.semantic_stats['entity_detections']['car_entities'] += 1
            elif 'student' in entity_name:
                self.semantic_stats['entity_detections']['student_entities'] += 1
            elif 'customer' in entity_name:
                self.semantic_stats['entity_detections']['customer_entities'] += 1
            elif 'product' in entity_name:
                self.semantic_stats['entity_detections']['product_entities'] += 1
            else:
                self.semantic_stats['entity_detections']['other_entities'] += 1
    
    def _calculate_confidence_score(self, analysis: Dict, enhancement_types: List[str]) -> float:
        """Calculate confidence score for the enhancement"""
        base_score = 0.5
        
        # Add points for detected metrics
        base_score += len(analysis['relevant_metrics']) * 0.1
        
        # Add points for detected dimensions
        base_score += len(analysis['relevant_dimensions']) * 0.1
        
        # Add points for detected entities
        base_score += len(analysis['relevant_entities']) * 0.1
        
        # Add points for applied enhancements
        base_score += len(enhancement_types) * 0.1
        
        # Bonus for complexity alignment
        complexity_score = analysis['complexity_score']
        if complexity_score > 0:
            base_score += min(complexity_score * 0.05, 0.2)
        
        return min(base_score, 1.0)
    
    def _simple_sql_generation(self, question: str, db_path: str) -> str:
        """Simple SQL generation fallback"""
        schema_info = self._get_database_schema(db_path)
        question_lower = question.lower()
        
        # Get first table as default
        tables = list(schema_info.keys())
        if not tables:
            return "SELECT 1"
        
        main_table = tables[0]
        
        # Simple pattern matching
        if any(word in question_lower for word in ['how many', 'count', 'number']):
            return f"SELECT COUNT(*) FROM {main_table}"
        elif any(word in question_lower for word in ['average', 'mean']):
            numeric_cols = self._find_relevant_columns(question, schema_info, 'numeric')
            if numeric_cols:
                col = numeric_cols[0].split('.')[1]  # Remove table prefix
                return f"SELECT AVG({col}) FROM {main_table}"
        elif any(word in question_lower for word in ['maximum', 'highest', 'max']):
            numeric_cols = self._find_relevant_columns(question, schema_info, 'numeric')
            if numeric_cols:
                col = numeric_cols[0].split('.')[1]  # Remove table prefix
                return f"SELECT MAX({col}) FROM {main_table}"
        elif any(word in question_lower for word in ['minimum', 'lowest', 'min']):
            numeric_cols = self._find_relevant_columns(question, schema_info, 'numeric')
            if numeric_cols:
                col = numeric_cols[0].split('.')[1]  # Remove table prefix
                return f"SELECT MIN({col}) FROM {main_table}"
        
        return f"SELECT * FROM {main_table}"
    
    def _get_database_schema(self, db_path: str) -> Dict[str, List[str]]:
        """Get database schema information"""
        schema_info = {}
        
        if not os.path.exists(db_path):
            return schema_info
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            # Get columns for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                schema_info[table] = columns
            
            conn.close()
            
        except Exception as e:
            print(f"Error getting schema: {e}")
        
        return schema_info
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze a question for semantic understanding without database context
        """
        if not self.semantic_enabled:
            return {'error': 'Semantic layer not available'}
        
        try:
            analysis = self.semantic_layer.analyze_query_intent(question)
            
            # Add some additional analysis
            enhanced_analysis = {
                **analysis,
                'readability_score': self._calculate_readability_score(question),
                'query_type': self._classify_query_type(analysis),
                'estimated_difficulty': self._estimate_query_difficulty(analysis)
            }
            
            return enhanced_analysis
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _calculate_readability_score(self, question: str) -> float:
        """Calculate how readable/clear the question is"""
        # Simple readability metric
        word_count = len(question.split())
        
        # Penalize very short or very long questions
        if word_count < 3:
            return 0.3
        elif word_count > 20:
            return 0.6
        else:
            return 0.8
    
    def _classify_query_type(self, analysis: Dict) -> str:
        """Classify the type of query based on analysis"""
        metrics = analysis['relevant_metrics']
        dimensions = analysis['relevant_dimensions']
        entities = analysis['relevant_entities']
        
        if not metrics and not dimensions:
            return "simple_retrieval"
        elif metrics and not dimensions:
            return "aggregation"
        elif metrics and dimensions:
            return "analytical"
        elif len(entities) > 1:
            return "multi_entity"
        else:
            return "standard"
    
    def _estimate_query_difficulty(self, analysis: Dict) -> str:
        """Estimate query difficulty for execution"""
        complexity_score = analysis['complexity_score']
        
        if complexity_score <= 2:
            return "easy"
        elif complexity_score <= 5:
            return "medium"
        elif complexity_score <= 8:
            return "hard"
        else:
            return "very_hard"
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive semantic enhancement statistics
        """
        stats = self.semantic_stats.copy()
        
        # Calculate derived statistics
        if stats['queries_analyzed'] > 0:
            stats['enhancement_rate'] = stats['queries_enhanced'] / stats['queries_analyzed'] * 100
            stats['avg_suggestions'] = stats['suggestions_made'] / stats['queries_analyzed']
            
            if stats['complexity_scores']:
                stats['avg_complexity'] = sum(stats['complexity_scores']) / len(stats['complexity_scores'])
                stats['max_complexity'] = max(stats['complexity_scores'])
                stats['min_complexity'] = min(stats['complexity_scores'])
        else:
            stats['enhancement_rate'] = 0
            stats['avg_suggestions'] = 0
            stats['avg_complexity'] = 0
            stats['max_complexity'] = 0
            stats['min_complexity'] = 0
        
        # Add semantic layer info
        if self.semantic_enabled:
            stats['semantic_layer_status'] = 'active'
            stats['total_metrics'] = len(self.semantic_layer.metrics)
            stats['total_dimensions'] = len(self.semantic_layer.dimensions)
            stats['total_entities'] = len(self.semantic_layer.entities)
        else:
            stats['semantic_layer_status'] = 'inactive'
            stats['total_metrics'] = 0
            stats['total_dimensions'] = 0
            stats['total_entities'] = 0
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self.semantic_stats = {
            'queries_analyzed': 0,
            'queries_enhanced': 0,
            'suggestions_made': 0,
            'complexity_scores': [],
            'enhancement_types': {
                'count_enhancements': 0,
                'aggregation_enhancements': 0,
                'grouping_enhancements': 0,
                'join_enhancements': 0,
                'filtering_enhancements': 0,
                'ordering_enhancements': 0
            },
            'entity_detections': {
                'car_entities': 0,
                'student_entities': 0,
                'customer_entities': 0,
                'product_entities': 0,
                'other_entities': 0
            }
        }
    
    def configure_enhancements(self, **kwargs):
        """Configure which enhancements are enabled"""
        for key, value in kwargs.items():
            if key in self.enhancement_config:
                self.enhancement_config[key] = value
    
    def get_enhancement_configuration(self) -> Dict[str, Any]:
        """Get current enhancement configuration"""
        return self.enhancement_config.copy()
    
    def batch_analyze_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Analyze a batch of questions for semantic understanding"""
        results = []
        
        for question in questions:
            analysis = self.analyze_question(question)
            results.append({
                'question': question,
                'analysis': analysis
            })
        
        return results
    
    def compare_sql_versions(self, question: str, db_path: str) -> Dict[str, Any]:
        """Compare original vs enhanced SQL versions"""
        if not self.semantic_enabled:
            return {'error': 'Semantic layer not available'}
        
        try:
            # Generate base SQL
            base_sql = self._generate_base_sql(question, db_path)
            
            # Generate enhanced SQL
            enhancement_result = self.generate_enhanced_sql(question, db_path)
            enhanced_sql = enhancement_result['enhanced_sql']
            
            # Compare versions
            comparison = {
                'question': question,
                'base_sql': base_sql,
                'enhanced_sql': enhanced_sql,
                'improvement_detected': base_sql != enhanced_sql,
                'enhancement_types': enhancement_result.get('enhancement_types', []),
                'confidence_score': enhancement_result.get('confidence_score', 0.0),
                'complexity_assessment': enhancement_result.get('complexity', 'unknown'),
                'suggestions': enhancement_result.get('suggestions', [])
            }
            
            # Calculate improvement metrics
            if comparison['improvement_detected']:
                comparison['sql_length_change'] = len(enhanced_sql) - len(base_sql)
                comparison['clause_additions'] = self._count_sql_clauses(enhanced_sql) - self._count_sql_clauses(base_sql)
            else:
                comparison['sql_length_change'] = 0
                comparison['clause_additions'] = 0
            
            return comparison
            
        except Exception as e:
            return {'error': f'Comparison failed: {str(e)}'}
    
    def _count_sql_clauses(self, sql: str) -> int:
        """Count SQL clauses in a query"""
        clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'JOIN']
        sql_upper = sql.upper()
        return sum(1 for clause in clauses if clause in sql_upper)
    
    def export_enhancement_report(self, output_path: str = None) -> str:
        """Export detailed enhancement report"""
        stats = self.get_semantic_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("SEMANTIC LAYER ENHANCEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {self._get_timestamp()}")
        report.append("")
        
        # Overall Statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 30)
        report.append(f"Queries Analyzed: {stats['queries_analyzed']}")
        report.append(f"Queries Enhanced: {stats['queries_enhanced']}")
        report.append(f"Enhancement Rate: {stats['enhancement_rate']:.1f}%")
        report.append(f"Average Suggestions per Query: {stats['avg_suggestions']:.1f}")
        report.append(f"Average Complexity Score: {stats['avg_complexity']:.1f}")
        report.append("")
        
        # Enhancement Types
        report.append("ENHANCEMENT TYPES")
        report.append("-" * 30)
        for enhancement_type, count in stats['enhancement_types'].items():
            report.append(f"{enhancement_type.replace('_', ' ').title()}: {count}")
        report.append("")
        
        # Entity Detections
        report.append("ENTITY DETECTIONS")
        report.append("-" * 30)
        for entity_type, count in stats['entity_detections'].items():
            report.append(f"{entity_type.replace('_', ' ').title()}: {count}")
        report.append("")
        
        # Semantic Layer Status
        report.append("SEMANTIC LAYER STATUS")
        report.append("-" * 30)
        report.append(f"Status: {stats['semantic_layer_status'].title()}")
        report.append(f"Total Metrics: {stats['total_metrics']}")
        report.append(f"Total Dimensions: {stats['total_dimensions']}")
        report.append(f"Total Entities: {stats['total_entities']}")
        report.append("")
        
        # Enhancement Configuration
        report.append("ENHANCEMENT CONFIGURATION")
        report.append("-" * 30)
        for config_key, config_value in self.enhancement_config.items():
            report.append(f"{config_key.replace('_', ' ').title()}: {config_value}")
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                print(f"Enhancement report saved to: {output_path}")
            except Exception as e:
                print(f"Failed to save report: {e}")
        
        return report_text
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reports"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Convenience function for easy integration
def evaluate_with_semantics(question: str, db_path: str, 
                          semantic_config_path: str = None,
                          use_existing_eval: bool = True) -> Dict[str, Any]:
    """
    Simple function to evaluate a question with semantic enhancement
    
    Args:
        question: Natural language question
        db_path: Path to SQLite database
        semantic_config_path: Path to semantic configuration file
        use_existing_eval: Whether to use existing evaluator if available
    
    Returns:
        Dictionary with SQL generation results and semantic analysis
    """
    evaluator = SemanticEvaluator(semantic_config_path=semantic_config_path)
    return evaluator.generate_enhanced_sql(question, db_path)

def batch_evaluate_with_semantics(questions: List[str], db_path: str,
                                 semantic_config_path: str = None) -> List[Dict[str, Any]]:
    """
    Evaluate multiple questions with semantic enhancement
    
    Args:
        questions: List of natural language questions
        db_path: Path to SQLite database
        semantic_config_path: Path to semantic configuration file
    
    Returns:
        List of evaluation results
    """
    evaluator = SemanticEvaluator(semantic_config_path=semantic_config_path)
    results = []
    
    for question in questions:
        result = evaluator.generate_enhanced_sql(question, db_path)
        results.append(result)
    
    return results

# Example usage and testing
def test_semantic_evaluator():
    """Test the semantic evaluator with example queries"""
    
    print("Testing Semantic Enhanced Evaluator")
    print("=" * 45)
    
    # Test questions
    test_questions = [
        "How many cars are in the database?",
        "What is the average horsepower by manufacturer?",
        "Show me the top 5 cars with highest MPG",
        "List all students enrolled in computer science courses",
        "What's the total revenue by region over time?"
    ]
    
    # Create evaluator
    evaluator = SemanticEvaluator()
    
    # Test each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i} ---")
        print(f"Question: {question}")
        
        # Analyze question semantically
        analysis = evaluator.analyze_question(question)
        
        if 'error' not in analysis:
            print(f"Complexity: {evaluator._assess_complexity(analysis)}")
            print(f"Query Type: {evaluator._classify_query_type(analysis)}")
            print(f"Difficulty: {evaluator._estimate_query_difficulty(analysis)}")
            print(f"Metrics found: {len(analysis.get('relevant_metrics', []))}")
            print(f"Dimensions found: {len(analysis.get('relevant_dimensions', []))}")
            print(f"Entities found: {len(analysis.get('relevant_entities', []))}")
        else:
            print(f"Error: {analysis['error']}")
        
        print("-" * 50)
    
    # Show final statistics
    stats = evaluator.get_semantic_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Queries Analyzed: {stats['queries_analyzed']}")
    print(f"  Semantic Layer Status: {stats['semantic_layer_status']}")
    print(f"  Total Metrics Available: {stats['total_metrics']}")
    print(f"  Total Dimensions Available: {stats['total_dimensions']}")
    print(f"  Total Entities Available: {stats['total_entities']}")
    
    return evaluator

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_evaluator = test_semantic_evaluator()
    
    # Export a sample report
    print(f"\nGenerating sample enhancement report...")
    report = test_evaluator.export_enhancement_report("semantic_enhancement_report.txt")
    print("Report preview:")
    print(report[:500] + "..." if len(report) > 500 else report)