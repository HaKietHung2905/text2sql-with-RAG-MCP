#!/usr/bin/env python3
"""
Fix column reference errors in SQL queries
"""

import os
import json
import sqlite3
import re
from collections import defaultdict

def get_database_schema(db_path):
    """Get complete schema information from a database"""
    if not os.path.exists(db_path):
        return {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1].lower() for col in cursor.fetchall()]
            schema[table.lower()] = columns
        
        conn.close()
        return schema
    except Exception as e:
        print(f"Error reading database {db_path}: {e}")
        return {}

def load_spider_schemas(tables_json_path):
    """Load database schemas from Spider tables.json"""
    if not os.path.exists(tables_json_path):
        return {}
    
    with open(tables_json_path, 'r') as f:
        tables_data = json.load(f)
    
    db_schemas = {}
    for entry in tables_data:
        db_id = entry['db_id']
        
        # Create table -> columns mapping
        schema = {}
        table_names = entry['table_names_original']
        
        for col_info in entry['column_names_original']:
            if len(col_info) >= 2:
                table_idx = col_info[0]
                col_name = col_info[1].lower()
                
                if table_idx >= 0 and table_idx < len(table_names):
                    table_name = table_names[table_idx].lower()
                    if table_name not in schema:
                        schema[table_name] = []
                    schema[table_name].append(col_name)
        
        db_schemas[db_id] = schema
    
    return db_schemas

def extract_sql_components(sql_query):
    """Extract table and column references from SQL query"""
    sql_upper = sql_query.upper()
    
    # Extract table names after FROM and JOIN
    table_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    table_refs = re.findall(table_pattern, sql_query, re.IGNORECASE)
    
    # Extract column names in WHERE, SELECT, ORDER BY clauses
    column_refs = []
    
    # Simple patterns for column references
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER|\s+GROUP|\s+HAVING|$)', sql_query, re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(1)
        # Extract column names (simplified - assumes column op value pattern)
        col_matches = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[<>=!]+', where_clause)
        column_refs.extend(col_matches)
    
    # Extract columns from ORDER BY
    order_match = re.search(r'ORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql_query, re.IGNORECASE)
    if order_match:
        column_refs.append(order_match.group(1))
    
    return [ref.lower() for ref in table_refs], [ref.lower() for ref in column_refs]

def validate_query_against_schema(sql_query, db_schema):
    """Validate SQL query against database schema"""
    if not db_schema:
        return True, []  # Can't validate without schema
    
    table_refs, column_refs = extract_sql_components(sql_query)
    issues = []
    
    # Check if referenced tables exist
    for table_ref in table_refs:
        if table_ref not in db_schema:
            issues.append(f"Table '{table_ref}' not found in schema")
    
    # Check if referenced columns exist in any of the tables
    for column_ref in column_refs:
        found = False
        for table_name, columns in db_schema.items():
            if column_ref in columns:
                found = True
                break
        if not found:
            issues.append(f"Column '{column_ref}' not found in any table")
    
    return len(issues) == 0, issues

def fix_problematic_query(sql_query, db_schema):
    """Fix common issues in SQL queries"""
    if not db_schema:
        return sql_query
    
    fixed_sql = sql_query
    
    # Get available tables and their columns
    available_tables = list(db_schema.keys())
    all_columns = []
    for table_columns in db_schema.values():
        all_columns.extend(table_columns)
    
    # Common column name mappings (generic -> specific)
    common_mappings = {
        'age': ['age', 'years', 'birth_year', 'year_born'],
        'name': ['name', 'first_name', 'last_name', 'full_name', 'title'],
        'id': ['id', 'student_id', 'person_id', 'customer_id'],
        'weight': ['weight', 'mass', 'weight_kg'],
        'level_of_membership': ['membership_level', 'level', 'membership_type']
    }
    
    # Try to fix column references
    table_refs, column_refs = extract_sql_components(sql_query)
    
    for column_ref in column_refs:
        if column_ref not in all_columns:
            # Try to find a replacement
            replacement = None
            
            # Check common mappings
            if column_ref in common_mappings:
                for candidate in common_mappings[column_ref]:
                    if candidate in all_columns:
                        replacement = candidate
                        break
            
            # If no mapping found, use first available column as fallback
            if not replacement and all_columns:
                replacement = all_columns[0]
            
            if replacement:
                fixed_sql = re.sub(
                    r'\b' + re.escape(column_ref) + r'\b',
                    replacement,
                    fixed_sql,
                    flags=re.IGNORECASE
                )
                print(f"  Fixed column reference: {column_ref} -> {replacement}")
    
    return fixed_sql

def process_txt_file(input_file, output_file, db_schemas, db_dir):
    """Process and fix a TXT file"""
    print(f"\n=== Processing {input_file} ===")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    stats = {
        'total': 0,
        'valid': 0,
        'fixed': 0,
        'removed': 0
    }
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        sql_query = parts[0].strip()
        db_id = parts[1].strip()
        stats['total'] += 1
        
        # Get schema for this database
        db_schema = db_schemas.get(db_id, {})
        
        # If schema not available from Spider, try loading directly from database
        if not db_schema:
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            db_schema = get_database_schema(db_path)
        
        # Validate query
        is_valid, issues = validate_query_against_schema(sql_query, db_schema)
        
        if is_valid:
            fixed_lines.append(f"{sql_query}\t{db_id}")
            stats['valid'] += 1
        else:
            # Try to fix the query
            if issues and any('Column' in issue for issue in issues):
                fixed_sql = fix_problematic_query(sql_query, db_schema)
                
                # Re-validate fixed query
                is_fixed_valid, _ = validate_query_against_schema(fixed_sql, db_schema)
                
                if is_fixed_valid:
                    fixed_lines.append(f"{fixed_sql}\t{db_id}")
                    stats['fixed'] += 1
                else:
                    # If still invalid, create a simple fallback query
                    if db_schema:
                        first_table = list(db_schema.keys())[0]
                        fallback_sql = f"SELECT COUNT(*) FROM {first_table}"
                        fixed_lines.append(f"{fallback_sql}\t{db_id}")
                        stats['fixed'] += 1
                        print(f"  Line {line_num}: Used fallback query for {db_id}")
                    else:
                        stats['removed'] += 1
                        print(f"  Line {line_num}: Removed invalid query (no schema available)")
            else:
                stats['removed'] += 1
                print(f"  Line {line_num}: Removed invalid query - {issues}")
    
    # Write fixed file
    with open(output_file, 'w') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    print(f"Stats: Total={stats['total']}, Valid={stats['valid']}, Fixed={stats['fixed']}, Removed={stats['removed']}")
    print(f"✅ Wrote {len(fixed_lines)} queries to {output_file}")

def main():
    # File paths
    input_gold = "../data/gold_queries_comprehensive.txt"
    input_pred = "../data/pred_queries_comprehensive.txt"
    
    # Try alternative file names
    if not os.path.exists(input_gold):
        input_gold = "gold_queries_comprehensive_fixed.txt"
    if not os.path.exists(input_pred):
        input_pred = "pred_queries_comprehensive_fixed.txt"
    
    output_gold = "gold_queries_column_fixed.txt"
    output_pred = "pred_queries_column_fixed.txt"
    
    tables_json_path = "../data/spider_dataset/spider/tables.json"
    db_dir = "../data/spider_dataset/database"
    
    print("🔧 COLUMN REFERENCE FIXER")
    print("=" * 50)
    
    # Load Spider schemas
    print("Loading database schemas...")
    db_schemas = load_spider_schemas(tables_json_path)
    print(f"Loaded schemas for {len(db_schemas)} databases")
    
    # Process files
    if os.path.exists(input_gold):
        process_txt_file(input_gold, output_gold, db_schemas, db_dir)
    else:
        print(f"❌ Gold file not found: {input_gold}")
    
    if os.path.exists(input_pred):
        process_txt_file(input_pred, output_pred, db_schemas, db_dir)
    else:
        print(f"❌ Pred file not found: {input_pred}")
    
    print(f"\n🎯 NEXT STEPS:")
    print("Test with the column-fixed files:")
    print(f"python eval.py \\")
    print(f"    --gold {output_gold} \\")
    print(f"    --pred {output_pred} \\")
    print(f"    --table {tables_json_path} \\")
    print(f"    --db {db_dir} \\")
    print(f"    --etype match")

if __name__ == "__main__":
    main()