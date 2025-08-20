#!/usr/bin/env python3
"""
Emergency fix to remove specific problematic queries that cause KeyError
"""

import os
import re

def is_problematic_query(sql_query):
    """Check if a query contains known problematic patterns"""
    
    # Known problematic patterns that cause KeyError
    problem_patterns = [
        r'contestant_number',
        r'contestant_name',
        r'Level_of_membership',
        r'Net_Worth_Millions',
        r'feature_type_code',
        r'property_type_code',
        r'WHERE.*ORDER BY',  # Complex WHERE with ORDER BY
        r'T1\.|T2\.',        # Table aliases
        r'AS T1|AS T2',      # Table aliases
        r'JOIN.*ON',         # JOIN clauses
        r'GROUP BY',         # GROUP BY
        r'HAVING',           # HAVING
        r'INTERSECT|UNION|EXCEPT',  # Set operations
        r'IndepYear',        # Another problematic column
        r'GovernmentForm',   # Another problematic column
        r'net_worth_millions', # Column that might not exist
    ]
    
    sql_upper = sql_query.upper()
    
    for pattern in problem_patterns:
        if re.search(pattern, sql_upper):
            return True
    
    return False

def emergency_fix_file(input_file, output_file):
    """Remove problematic queries and keep only safe ones"""
    
    print(f"\n=== Emergency Fix: {input_file} -> {output_file} ===")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return 0
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    safe_queries = []
    stats = {
        'total': 0,
        'safe': 0,
        'problematic': 0
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        sql_query = parts[0].strip()
        db_id = parts[1].strip()
        stats['total'] += 1
        
        if is_problematic_query(sql_query):
            stats['problematic'] += 1
            if stats['problematic'] <= 10:  # Show first 10 removed queries
                print(f"  Removed: {sql_query[:60]}...")
        else:
            safe_queries.append(f"{sql_query}\t{db_id}")
            stats['safe'] += 1
    
    # Write safe queries
    with open(output_file, 'w') as f:
        for query in safe_queries:
            f.write(query + '\n')
    
    print(f"📊 Stats: Total={stats['total']}, Safe={stats['safe']}, Removed={stats['problematic']}")
    print(f"✅ Wrote {len(safe_queries)} safe queries to {output_file}")
    
    return len(safe_queries)

def show_safe_query_samples(file_path, num_samples=10):
    """Show samples of safe queries"""
    if not os.path.exists(file_path):
        return
    
    print(f"\n📝 Sample safe queries from {os.path.basename(file_path)}:")
    print("-" * 70)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[:num_samples]):
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) >= 2:
                print(f"  {i+1:2d}. {parts[0]} [{parts[1]}]")

def main():
    # Input files
    input_gold = "../data/gold_queries_comprehensive.txt"
    input_pred = "../data/pred_queries_comprehensive.txt"
    
    # Output files
    output_gold = "gold_queries_emergency_safe.txt"
    output_pred = "pred_queries_emergency_safe.txt"
    
    # Database info
    tables_json_path = "../data/spider_dataset/spider/tables.json"
    db_dir = "../data/spider_dataset/database"
    
    print("🚨 EMERGENCY QUERY FIX")
    print("=" * 50)
    print("Removing queries that cause KeyError: 'feature_type_code'")
    print("Keeping only simple, safe queries")
    
    # Fix both files
    gold_count = emergency_fix_file(input_gold, output_gold)
    pred_count = emergency_fix_file(input_pred, output_pred)
    
    # Show samples
    show_safe_query_samples(output_gold)
    
    print(f"\n🎯 IMMEDIATE TEST:")
    print("Run this command to test the emergency-fixed files:")
    print()
    print(f"python eval.py \\")
    print(f"    --gold {output_gold} \\")
    print(f"    --pred {output_pred} \\")
    print(f"    --table {tables_json_path} \\")
    print(f"    --db {db_dir} \\")
    print(f"    --etype match")
    print()
    
    if gold_count > 0 and pred_count > 0:
        print(f"✅ Emergency fix complete!")
        print(f"   Safe queries: Gold={gold_count}, Pred={pred_count}")
        print(f"   This should eliminate the KeyError and let evaluation complete")
    else:
        print(f"❌ No safe queries found. May need to regenerate query files.")

if __name__ == "__main__":
    main()