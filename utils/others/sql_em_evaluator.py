#!/usr/bin/env python3
"""
Standalone SQL Exact Match (EM) Accuracy Evaluator
Extracts and runs only the exact match evaluation logic from the original evaluation script.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Add current directory and data directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(data_dir) 

try:
    from process_sql import get_schema, Schema, get_sql
except ImportError:
    print("Error: Cannot import process_sql. Please ensure the process_sql module is available.")
    print("You may need to adjust the sys.path.append() line at the top of this script.")
    sys.exit(1)

# Constants from original evaluation script
DISABLE_VALUE = True
DISABLE_DISTINCT = True

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

def find_spider_dataset():
    """Find the Spider dataset in the project structure"""
    possible_paths = [
        os.path.join('data', 'spider_dataset', 'spider'),
        os.path.join('..', 'data', 'spider_dataset', 'spider'),
        os.path.join('data', 'spider_dataset'),
        os.path.join('..', 'data', 'spider_dataset'),
        'spider_dataset/spider',
        '../spider_dataset/spider'
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            tables_json = os.path.join(abs_path, 'tables.json')
            if os.path.exists(tables_json):
                print(f"✅ Found Spider dataset at: {abs_path}")
                return abs_path
    
    print("❌ Spider dataset not found in expected locations")
    return None

def find_spider_databases():
    """Find the Spider database directory"""
    spider_dir = find_spider_dataset()
    if not spider_dir:
        return None
    
    # Common database directory locations
    possible_db_dirs = [
        os.path.join(spider_dir, 'database'),
        os.path.join(spider_dir, '..', 'database'),
        os.path.join(spider_dir, '..', '..', 'database'),
        os.path.join(os.path.dirname(spider_dir), 'database'),
    ]
    
    for db_dir in possible_db_dirs:
        abs_db_dir = os.path.abspath(db_dir)
        if os.path.exists(abs_db_dir):
            print(f"✅ Found database directory at: {abs_db_dir}")
            return abs_db_dir
    
    print("❌ Database directory not found")
    return None

def get_scores(count, pred_total, label_total):
    """Calculate precision, recall, and F1 scores"""
    if pred_total != label_total:
        return 0, 0, 0
    elif count == pred_total:
        return 1, 1, 1
    return 0, 0, 0

def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    """Rebuild column unit with foreign key mapping"""
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct

def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    """Rebuild value unit with foreign key mapping"""
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2

def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    """Rebuild table unit with foreign key mapping"""
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql

def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    """Rebuild condition unit with foreign key mapping"""
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return not_op, op_id, val_unit, val1, val2

def rebuild_condition_col(valid_col_units, condition, kmap):
    """Rebuild condition with foreign key mapping"""
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition

def rebuild_from_col(valid_col_units, from_, kmap):
    """Rebuild FROM clause with foreign key mapping"""
    if from_ is None:
        return from_

    from_['table_units'] = [rebuild_table_unit_col(valid_col_units, table_unit, kmap) for table_unit in from_['table_units']]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    return from_

def rebuild_group_by_col(valid_col_units, group_by, kmap):
    """Rebuild GROUP BY clause with foreign key mapping"""
    if group_by is None:
        return group_by
    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]

def rebuild_order_by_col(valid_col_units, order_by, kmap):
    """Rebuild ORDER BY clause with foreign key mapping"""
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units]
    return direction, new_val_units

def rebuild_select_col(valid_col_units, sel, kmap):
    """Rebuild SELECT clause with foreign key mapping"""
    if sel is None:
        return sel
    distinct, _list = sel
    new_list = []
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    if DISABLE_DISTINCT:
        distinct = None
    return distinct, new_list

def rebuild_sql_col(valid_col_units, sql, kmap):
    """Rebuild entire SQL with foreign key mapping"""
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap)
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'], kmap)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_condition_col(valid_col_units, sql['having'], kmap)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap)

    return sql

def rebuild_cond_unit_val(cond_unit):
    """Rebuild condition unit for value evaluation"""
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2

def rebuild_condition_val(condition):
    """Rebuild condition for value evaluation"""
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res

def rebuild_sql_val(sql):
    """Rebuild SQL for value evaluation"""
    if sql is None or not DISABLE_VALUE:
        return sql

    sql['from']['conds'] = rebuild_condition_val(sql['from']['conds'])
    sql['having'] = rebuild_condition_val(sql['having'])
    sql['where'] = rebuild_condition_val(sql['where'])
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])

    return sql

def build_valid_col_units(table_units, schema):
    """Build valid column units for foreign key evaluation"""
    col_ids = [table_unit[1] for table_unit in table_units if table_unit[0] == TABLE_TYPE['table_unit']]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units = []
    for value in schema.idMap.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units

def eval_sel(pred, label):
    """Evaluate SELECT clause"""
    pred_sel = pred['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg

def eval_where(pred, label):
    """Evaluate WHERE clause"""
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg

def eval_group(pred, label):
    """Evaluate GROUP BY clause"""
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred.split(".")[1] if "." in pred else pred for pred in pred_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt

def eval_having(pred, label):
    """Evaluate HAVING clause"""
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt

def eval_order(pred, label):
    """Evaluate ORDER BY clause"""
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and \
            ((pred['limit'] is None and label['limit'] is None) or (pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt

def eval_and_or(pred, label):
    """Evaluate AND/OR operators"""
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1, 1, 1
    return len(pred_ao), len(label_ao), 0

def eval_nested(pred, label):
    """Evaluate nested queries"""
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        evaluator = Evaluator()
        cnt += evaluator.eval_exact_match(pred, label)
    return label_total, pred_total, cnt

def eval_IUEN(pred, label):
    """Evaluate INTERSECT, UNION, EXCEPT"""
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], label['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], label['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], label['union'])
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt

def get_keywords(sql):
    """Extract keywords from SQL"""
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res

def eval_keywords(pred, label):
    """Evaluate SQL keywords"""
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt

class Evaluator:
    """SQL Exact Match Evaluator"""
    
    def __init__(self):
        self.partial_scores = None

    def eval_exact_match(self, pred, label):
        """Evaluate exact match between predicted and label SQL"""
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for key, score in partial_scores.items():
            if score['f1'] != 1:
                return 0

        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return label_tables == pred_tables
        return 1

    def eval_partial_match(self, pred, label):
        """Evaluate partial matches for different SQL components"""
        res = {}

        # Evaluate SELECT
        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        # Evaluate WHERE
        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        # Evaluate GROUP BY
        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        # Evaluate HAVING
        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        # Evaluate ORDER BY
        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        # Evaluate AND/OR
        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        # Evaluate INTERSECT/UNION/EXCEPT
        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        # Evaluate keywords
        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total, 'pred_total': pred_total}

        return res

def build_foreign_key_map(entry):
    """Build foreign key mapping from table schema entry"""
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = []
    for col_orig in cols_orig:
        # Handle different formats in Spider dataset
        if isinstance(col_orig, list) and len(col_orig) == 2:
            table_idx, col_name = col_orig
            # Check if table_idx is a valid integer and >= 0
            if isinstance(table_idx, int) and table_idx >= 0:
                t = tables_orig[table_idx]
                c = col_name
                cols.append("__" + t.lower() + "." + c.lower() + "__")
            else:
                # Handle special cases like [-1, "*"] or ["*"]
                cols.append("__all__")
        elif isinstance(col_orig, list) and len(col_orig) == 1:
            # Handle cases like ["*"]
            cols.append("__all__")
        else:
            # Fallback for unexpected formats
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map

def build_foreign_key_map_from_json(table_file):
    """Build foreign key mappings from tables.json file"""
    with open(table_file) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables

def evaluate_exact_match(gold_file, pred_file, db_dir, table_file, verbose=False):
    """
    Evaluate exact match accuracy between predicted and gold SQL queries
    
    Args:
        gold_file: Path to file containing gold SQL queries
        pred_file: Path to file containing predicted SQL queries
        db_dir: Directory containing database files (can be None for auto-detection)
        table_file: Path to tables.json schema file (can be None for auto-detection)
        verbose: Whether to print detailed results for failed cases
    
    Returns:
        dict: Evaluation results with exact match accuracy
    """
    
    # Auto-detect Spider dataset if paths not provided
    if table_file is None:
        spider_dir = find_spider_dataset()
        if spider_dir:
            table_file = os.path.join(spider_dir, 'tables.json')
        else:
            raise ValueError("Could not find Spider dataset. Please specify --table argument.")
    
    if db_dir is None:
        db_dir = find_spider_databases()
        if not db_dir:
            raise ValueError("Could not find database directory. Please specify --db argument.")
    
    print(f"Using tables.json: {table_file}")
    print(f"Using database directory: {db_dir}")
    
    # Load foreign key mappings
    kmaps = build_foreign_key_map_from_json(table_file)
    
    # Load gold queries
    with open(gold_file) as f:
        glist = []
        gseq_one = []
        for l in f.readlines():
            if len(l.strip()) == 0:
                glist.append(gseq_one)
                gseq_one = []
            else:
                lstrip = l.strip().split('\t')
                gseq_one.append(lstrip)
        if len(gseq_one) != 0:
            glist.append(gseq_one)

    # Load predicted queries
    with open(pred_file) as f:
        plist = []
        pseq_one = []
        for l in f.readlines():
            if len(l.strip()) == 0:
                plist.append(pseq_one)
                pseq_one = []
            else:
                pseq_one.append(l.strip().split('\t'))
        if len(pseq_one) != 0:
            plist.append(pseq_one)

    assert len(plist) == len(glist), f"Number of prediction sessions ({len(plist)}) must equal gold sessions ({len(glist)})"

    evaluator = Evaluator()
    
    # Initialize counters
    total_count = 0
    exact_match_count = 0
    failed_cases = []

    # Evaluate each query pair
    for i, (p_session, g_session) in enumerate(tqdm(zip(plist, glist), desc="Evaluating Exact Match")):
        for idx, (p, g) in enumerate(zip(p_session, g_session)):
            p_str = p[0]
            g_str, db_name = g
            
            total_count += 1
            
            # Get database schema
            db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
            if not os.path.exists(db_path):
                if verbose:
                    print(f"Warning: Database file not found: {db_path}")
                continue
                
            schema = Schema(get_schema(db_path))
            
            # Parse gold SQL
            print("db_path", db_path) 
            try:
                g_sql = get_sql(schema, g_str)
            except Exception as e:
                if verbose:
                    print(f"Error parsing gold SQL at {i}-{idx}: {e}")
                continue
            
            # Parse predicted SQL
            try:
                p_sql = get_sql(schema, p_str)
            except Exception as e:
                if verbose:
                    print(f"Error parsing predicted SQL at {i}-{idx}: {e}")
                # Use empty SQL structure for invalid predictions
                p_sql = {
                    "except": None,
                    "from": {"conds": [], "table_units": []},
                    "groupBy": [],
                    "having": [],
                    "intersect": None,
                    "limit": None,
                    "orderBy": [],
                    "select": [False, []],
                    "union": None,
                    "where": []
                }
            
            # Rebuild SQL with foreign key mapping and value handling
            kmap = kmaps[db_name]
            
            # Process gold SQL
            g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
            g_sql = rebuild_sql_val(g_sql)
            g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
            
            # Process predicted SQL
            p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
            p_sql = rebuild_sql_val(p_sql)
            p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)
            
            # Evaluate exact match
            exact_score = evaluator.eval_exact_match(p_sql, g_sql)
            
            if exact_score == 1:
                exact_match_count += 1
            else:
                failed_case = {
                    'id': f"{i}-{idx}",
                    'predicted': p_str,
                    'gold': g_str,
                    'database': db_name,
                    'partial_scores': evaluator.partial_scores
                }
                failed_cases.append(failed_case)
                
                if verbose:
                    print(f"\nFailed case {i}-{idx}:")
                    print(f"Predicted: {p_str}")
                    print(f"Gold: {g_str}")
                    print(f"Database: {db_name}")
                    print("Partial scores:", evaluator.partial_scores)

    # Calculate final accuracy
    exact_accuracy = exact_match_count / total_count if total_count > 0 else 0
    
    results = {
        'exact_match_accuracy': exact_accuracy,
        'total_queries': total_count,
        'correct_queries': exact_match_count,
        'failed_cases': failed_cases
    }
    
    print(f"\n=== EXACT MATCH EVALUATION RESULTS ===")
    print(f"Total queries: {total_count}")
    print(f"Correct queries: {exact_match_count}")
    print(f"Exact Match Accuracy: {exact_accuracy:.4f} ({exact_accuracy*100:.2f}%)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="SQL Exact Match Accuracy Evaluator")
    parser.add_argument('--gold', required=True, help="Path to gold SQL queries file")
    parser.add_argument('--pred', required=True, help="Path to predicted SQL queries file")
    parser.add_argument('--db', required=True, help="Directory containing database files")
    parser.add_argument('--table', required=True, help="Path to tables.json schema file")
    parser.add_argument('--verbose', action='store_true', help="Print detailed results for failed cases")
    parser.add_argument('--output', help="Path to save detailed results as JSON")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_exact_match(
        gold_file=args.gold,
        pred_file=args.pred, 
        db_dir=args.db,
        table_file=args.table,
        verbose=args.verbose
    )
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {args.output}")

if __name__ == "__main__":
    main()