#!/usr/bin/env python3
"""
Create SQLite database files for Spider dataset based on tables.json schema
"""

import json
import os
import sqlite3
from pathlib import Path

def load_spider_tables(tables_json_path):
    """Load Spider tables.json file"""
    with open(tables_json_path, 'r') as f:
        return json.load(f)

def get_sql_type(spider_type):
    """Convert Spider column type to SQLite type"""
    type_mapping = {
        'text': 'TEXT',
        'number': 'INTEGER', 
        'boolean': 'INTEGER',
        'time': 'TEXT',
        'others': 'TEXT'
    }
    return type_mapping.get(spider_type.lower(), 'TEXT')

def create_database_from_schema(db_entry, db_path):
    """Create a SQLite database from Spider schema entry"""
    db_id = db_entry['db_id']
    table_names = db_entry.get('table_names_original', [])
    column_names = db_entry.get('column_names_original', [])
    column_types = db_entry.get('column_types', [])
    foreign_keys = db_entry.get('foreign_keys', [])
    primary_keys = db_entry.get('primary_keys', [])
    
    print(f"Creating database: {db_id}")
    print(f"  Tables: {len(table_names)}")
    print(f"  Columns: {len(column_names)}")
    
    # Create database directory
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Skip the first column entry if it's the "*" wildcard
        start_idx = 1 if column_names and column_names[0] == ['*'] else 0
        columns = column_names[start_idx:]
        types = column_types[start_idx:] if column_types else []
        
        # Create tables
        for table_idx, table_name in enumerate(table_names):
            print(f"  Creating table: {table_name}")
            
            # Find columns for this table
            table_columns = []
            for col_idx, col_info in enumerate(columns):
                if isinstance(col_info, list) and len(col_info) >= 2:
                    col_table_idx = col_info[0]
                    col_name = col_info[1]
                    
                    # Check if this column belongs to current table
                    if col_table_idx == table_idx:
                        col_type = get_sql_type(types[col_idx]) if col_idx < len(types) else 'TEXT'
                        
                        # Check if this is a primary key
                        is_primary = (start_idx + col_idx) in primary_keys
                        
                        table_columns.append({
                            'name': col_name,
                            'type': col_type,
                            'is_primary': is_primary
                        })
            
            # If no columns found, create a basic table
            if not table_columns:
                table_columns = [
                    {'name': 'id', 'type': 'INTEGER', 'is_primary': True},
                    {'name': 'name', 'type': 'TEXT', 'is_primary': False}
                ]
            
            # Build CREATE TABLE statement
            column_defs = []
            for col in table_columns:
                col_def = f"{col['name']} {col['type']}"
                if col['is_primary']:
                    col_def += " PRIMARY KEY"
                column_defs.append(col_def)
            
            create_sql = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
            
            try:
                cursor.execute(create_sql)
                print(f"    ✅ Created with {len(table_columns)} columns")
                
                # Insert sample data
                insert_sample_data(cursor, table_name, table_columns)
                
            except Exception as e:
                print(f"    ❌ Error creating table {table_name}: {e}")
        
        # Add foreign key constraints (if needed)
        add_foreign_keys(cursor, table_names, columns, foreign_keys, start_idx)
        
        conn.commit()
        print(f"✅ Successfully created database: {db_path}")
        
    except Exception as e:
        print(f"❌ Error creating database {db_id}: {e}")
        
    finally:
        conn.close()

def insert_sample_data(cursor, table_name, columns):
    """Insert sample data into table"""
    try:
        # Generate sample values
        sample_values = []
        for col in columns:
            if col['type'] == 'INTEGER':
                if col['is_primary']:
                    sample_values.append('1')
                else:
                    sample_values.append('100')
            else:  # TEXT
                sample_values.append(f"'sample_{col['name']}'")
        
        # Insert sample row
        placeholders = ', '.join(['?'] * len(columns))
        values = []
        for col in columns:
            if col['type'] == 'INTEGER':
                values.append(1 if col['is_primary'] else 100)
            else:
                values.append(f"sample_{col['name']}")
        
        insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
        cursor.execute(insert_sql, values)
        
        # Insert a few more sample rows
        if len(columns) > 0:
            for i in range(2, 4):  # Add 2 more rows
                values = []
                for col in columns:
                    if col['type'] == 'INTEGER':
                        values.append(i if col['is_primary'] else 100 + i)
                    else:
                        values.append(f"sample_{col['name']}_{i}")
                
                try:
                    cursor.execute(insert_sql, values)
                except:
                    pass  # Skip if duplicate primary key
        
    except Exception as e:
        print(f"    Warning: Could not insert sample data: {e}")

def add_foreign_keys(cursor, table_names, columns, foreign_keys, start_idx):
    """Add foreign key information (SQLite doesn't enforce FKs by default)"""
    # SQLite foreign keys are defined in CREATE TABLE, 
    # so we'll just log them for reference
    if foreign_keys:
        print(f"  Foreign keys defined: {len(foreign_keys)}")

def create_all_spider_databases(tables_json_path, output_dir):
    """Create all databases from Spider tables.json"""
    spider_tables = load_spider_tables(tables_json_path)
    
    print(f"Found {len(spider_tables)} databases in Spider dataset")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    created_count = 0
    failed_count = 0
    
    for db_entry in spider_tables:
        db_id = db_entry['db_id']
        db_path = os.path.join(output_dir, db_id, f"{db_id}.sqlite")
        
        try:
            create_database_from_schema(db_entry, db_path)
            created_count += 1
        except Exception as e:
            print(f"❌ Failed to create {db_id}: {e}")
            failed_count += 1
        
        print()  # Empty line between databases
    
    print("=" * 60)
    print(f"✅ Successfully created: {created_count} databases")
    print(f"❌ Failed to create: {failed_count} databases")
    print(f"📁 Databases location: {os.path.abspath(output_dir)}")

def check_existing_databases(output_dir, spider_tables):
    """Check which databases already exist"""
    existing = []
    missing = []
    
    for db_entry in spider_tables:
        db_id = db_entry['db_id']
        db_path = os.path.join(output_dir, db_id, f"{db_id}.sqlite")
        
        if os.path.exists(db_path):
            existing.append(db_id)
        else:
            missing.append(db_id)
    
    return existing, missing

def verify_database(db_path):
    """Verify that database was created correctly"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return len(tables) > 0, tables
        
    except Exception as e:
        return False, []

def main():
    # Configuration
    tables_json_path = "../data/spider_dataset/spider/tables.json"
    output_dir = "../data/spider_dataset/database"
    
    print("🕷️  SPIDER DATABASE CREATOR")
    print("=" * 50)
    
    # Check if tables.json exists
    if not os.path.exists(tables_json_path):
        print(f"❌ tables.json not found at: {tables_json_path}")
        print("Please update the path to your Spider dataset")
        return
    
    # Load Spider schema
    spider_tables = load_spider_tables(tables_json_path)
    
    # Check existing databases
    existing, missing = check_existing_databases(output_dir, spider_tables)
    
    print(f"📊 Database Status:")
    print(f"  Total databases: {len(spider_tables)}")
    print(f"  Already exist: {len(existing)}")
    print(f"  Need to create: {len(missing)}")
    print()
    
    if len(missing) == 0:
        print("✅ All databases already exist!")
        
        # Verify a few databases
        print("\n🔍 Verifying some databases...")
        for db_id in existing[:5]:  # Check first 5
            db_path = os.path.join(output_dir, db_id, f"{db_id}.sqlite")
            is_valid, tables = verify_database(db_path)
            status = "✅" if is_valid else "❌"
            print(f"  {status} {db_id}: {len(tables)} tables")
        
        return
    
    # Ask user confirmation
    print(f"About to create {len(missing)} databases...")
    response = input("Continue? (y/N): ").lower().strip()
    
    if response != 'y':
        print("Cancelled.")
        return
    
    # Create missing databases
    missing_entries = [entry for entry in spider_tables if entry['db_id'] in missing]
    
    print(f"\n🚀 Creating {len(missing_entries)} databases...")
    print("=" * 60)
    
    created_count = 0
    for db_entry in missing_entries:
        db_id = db_entry['db_id']
        db_path = os.path.join(output_dir, db_id, f"{db_id}.sqlite")
        
        try:
            create_database_from_schema(db_entry, db_path)
            created_count += 1
        except Exception as e:
            print(f"❌ Failed to create {db_id}: {e}")
    
    print("=" * 60)
    print(f"🎉 Created {created_count} new databases!")
    print(f"📁 Location: {os.path.abspath(output_dir)}")
    print("\n🎯 Next steps:")
    print("1. Run your SQL evaluator:")
    print("   python sql_em_evaluator.py --gold gold_queries.txt --pred pred_queries.txt --verbose")

if __name__ == "__main__":
    main()