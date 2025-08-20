"""
Dataset utilities for Text2SQL Spider dataset.
Includes functions for loading, splitting, and analyzing the data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter
import random
from dataclasses import dataclass

@dataclass
class DatasetSplit:
    """Container for dataset splits."""
    train: List[Dict]
    dev: List[Dict]
    test: Optional[List[Dict]] = None
    
    def __len__(self):
        return len(self.train) + len(self.dev) + (len(self.test) if self.test else 0)
    
    def summary(self):
        """Print summary of the split."""
        print(f"Dataset Split Summary:")
        print(f"  Train: {len(self.train)} examples")
        print(f"  Dev: {len(self.dev)} examples")
        if self.test:
            print(f"  Test: {len(self.test)} examples")
        print(f"  Total: {len(self)} examples")

class SpiderDatasetSplitter:
    """
    Utility class for loading and splitting Spider dataset in various ways.
    Supports different splitting strategies for Text2SQL experiments.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.spider_dir = self.data_dir / "spider"
        
    def load_raw_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load raw Spider train and dev data."""
        train_path = self.spider_dir / "train_spider.json"
        dev_path = self.spider_dir / "dev.json"
        
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            
        with open(dev_path, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
            
        print(f"Loaded {len(train_data)} train examples, {len(dev_data)} dev examples")
        return train_data, dev_data
    
    def random_split(self, 
                    train_ratio: float = 0.7, 
                    dev_ratio: float = 0.15, 
                    test_ratio: float = 0.15,
                    seed: int = 42) -> DatasetSplit:
        """
        Random split of the combined dataset.
        
        Args:
            train_ratio: Proportion for training
            dev_ratio: Proportion for development/validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        train_data, dev_data = self.load_raw_data()
        all_data = train_data + dev_data
        
        # Shuffle data
        random.seed(seed)
        random.shuffle(all_data)
        
        n_total = len(all_data)
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)
        
        split = DatasetSplit(
            train=all_data[:n_train],
            dev=all_data[n_train:n_train+n_dev],
            test=all_data[n_train+n_dev:] if test_ratio > 0 else None
        )
        
        print(f"Random split with seed {seed}:")
        split.summary()
        return split
    
    def database_split(self, 
                      dev_databases: Optional[List[str]] = None,
                      test_databases: Optional[List[str]] = None) -> DatasetSplit:
        """
        Split by database - ensures no database appears in multiple splits.
        This is the standard Spider evaluation setup.
        
        Args:
            dev_databases: List of database IDs for dev set
            test_databases: List of database IDs for test set
        """
        train_data, original_dev_data = self.load_raw_data()
        
        # Get all unique databases
        all_databases = set()
        for item in train_data + original_dev_data:
            all_databases.add(item['db_id'])
        
        print(f"Found {len(all_databases)} unique databases")
        
        # If not specified, use the original Spider split
        if dev_databases is None:
            dev_databases = list(set(item['db_id'] for item in original_dev_data))
        
        # Group data by database
        db_to_examples = defaultdict(list)
        for item in train_data + original_dev_data:
            db_to_examples[item['db_id']].append(item)
        
        # Create splits
        new_train = []
        new_dev = []
        new_test = []
        
        for db_id, examples in db_to_examples.items():
            if test_databases and db_id in test_databases:
                new_test.extend(examples)
            elif db_id in dev_databases:
                new_dev.extend(examples)
            else:
                new_train.extend(examples)
        
        split = DatasetSplit(
            train=new_train,
            dev=new_dev,
            test=new_test if new_test else None
        )
        
        print(f"Database split:")
        print(f"  Train databases: {len([db for db in all_databases if db not in dev_databases and (not test_databases or db not in test_databases)])}")
        print(f"  Dev databases: {len(dev_databases)}")
        if test_databases:
            print(f"  Test databases: {len(test_databases)}")
        split.summary()
        
        return split
    
    def difficulty_split(self, 
                        difficulty_levels: Dict[str, List[str]] = None) -> Dict[str, DatasetSplit]:
        """
        Split data by SQL difficulty/complexity.
        
        Args:
            difficulty_levels: Dict mapping difficulty names to SQL keywords/patterns
        """
        if difficulty_levels is None:
            difficulty_levels = {
                "easy": ["SELECT", "FROM", "WHERE"],
                "medium": ["JOIN", "GROUP BY", "ORDER BY", "HAVING"],
                "hard": ["NESTED", "UNION", "INTERSECT", "EXCEPT", "SUBQUERY"]
            }
        
        train_data, dev_data = self.load_raw_data()
        all_data = train_data + dev_data
        
        # Classify examples by difficulty
        difficulty_data = defaultdict(list)
        
        for item in all_data:
            sql = item['sql'].upper()
            
            # Check for difficulty markers
            is_hard = any(keyword in sql for keyword in difficulty_levels["hard"]) or \
                     sql.count('SELECT') > 1  # Multiple SELECT = nested/subquery
            
            is_medium = any(keyword in sql for keyword in difficulty_levels["medium"])
            
            if is_hard:
                difficulty_data["hard"].append(item)
            elif is_medium:
                difficulty_data["medium"].append(item)
            else:
                difficulty_data["easy"].append(item)
        
        # Create splits for each difficulty
        splits = {}
        for difficulty, examples in difficulty_data.items():
            # 80-20 split for each difficulty
            random.shuffle(examples)
            n_train = int(len(examples) * 0.8)
            
            splits[difficulty] = DatasetSplit(
                train=examples[:n_train],
                dev=examples[n_train:],
                test=None
            )
            
            print(f"\n{difficulty.title()} difficulty:")
            splits[difficulty].summary()
        
        return splits
    
    def few_shot_split(self, 
                      n_shot: int = 5, 
                      n_eval: int = 100,
                      strategy: str = "random") -> DatasetSplit:
        """
        Create splits for few-shot learning experiments.
        
        Args:
            n_shot: Number of examples for few-shot learning
            n_eval: Number of examples for evaluation
            strategy: How to select examples ("random", "diverse", "similar")
        """
        train_data, dev_data = self.load_raw_data()
        
        if strategy == "random":
            # Random selection
            all_examples = train_data + dev_data
            random.shuffle(all_examples)
            
            few_shot_examples = all_examples[:n_shot]
            eval_examples = all_examples[n_shot:n_shot+n_eval]
            remaining = all_examples[n_shot+n_eval:]
            
        elif strategy == "diverse":
            # Select diverse examples (different databases)
            db_to_examples = defaultdict(list)
            for item in train_data:
                db_to_examples[item['db_id']].append(item)
            
            few_shot_examples = []
            used_dbs = set()
            
            # Try to get one example from each database
            for db_id, examples in db_to_examples.items():
                if len(few_shot_examples) < n_shot:
                    few_shot_examples.append(random.choice(examples))
                    used_dbs.add(db_id)
            
            # Fill remaining with random examples
            remaining_examples = [item for item in train_data if item['db_id'] not in used_dbs]
            while len(few_shot_examples) < n_shot and remaining_examples:
                few_shot_examples.append(remaining_examples.pop())
            
            eval_examples = dev_data[:n_eval]
            remaining = [item for item in train_data if item not in few_shot_examples]
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        split = DatasetSplit(
            train=few_shot_examples,
            dev=eval_examples,
            test=remaining if remaining else None
        )
        
        print(f"Few-shot split ({strategy}):")
        split.summary()
        return split
    
    def cross_domain_split(self, domains: Optional[List[str]] = None) -> Dict[str, DatasetSplit]:
        """
        Split data by domain for cross-domain evaluation.
        """
        train_data, dev_data = self.load_raw_data()
        
        # Define domains based on database names (simplified)
        if domains is None:
            domains = ["academic", "music", "sports", "business", "government"]
        
        domain_keywords = {
            "academic": ["college", "school", "student", "course", "university"],
            "music": ["singer", "song", "concert", "album", "artist"],
            "sports": ["player", "team", "game", "match", "tournament"],
            "business": ["employee", "company", "customer", "order", "product"],
            "government": ["city", "country", "state", "department", "official"]
        }
        
        # Classify databases by domain
        domain_data = defaultdict(list)
        unclassified = []
        
        for item in train_data + dev_data:
            db_id = item['db_id'].lower()
            classified = False
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in db_id for keyword in keywords):
                    domain_data[domain].append(item)
                    classified = True
                    break
            
            if not classified:
                unclassified.append(item)
        
        # Create splits for each domain
        splits = {}
        for domain, examples in domain_data.items():
            if len(examples) > 10:  # Only create split if enough examples
                random.shuffle(examples)
                n_train = int(len(examples) * 0.7)
                
                splits[domain] = DatasetSplit(
                    train=examples[:n_train],
                    dev=examples[n_train:],
                    test=None
                )
                
                print(f"\n{domain.title()} domain:")
                splits[domain].summary()
        
        if unclassified:
            print(f"\nUnclassified: {len(unclassified)} examples")
        
        return splits
    
    def analyze_dataset(self) -> Dict:
        """Analyze the dataset and return statistics."""
        train_data, dev_data = self.load_raw_data()
        all_data = train_data + dev_data
        
        # Basic statistics
        stats = {
            "total_examples": len(all_data),
            "train_examples": len(train_data),
            "dev_examples": len(dev_data),
            "unique_databases": len(set(item['db_id'] for item in all_data)),
            "avg_question_length": np.mean([len(item['question'].split()) for item in all_data]),
            "avg_sql_length": np.mean([len(item['sql'].split()) for item in all_data])
        }
        
        # Database distribution
        db_counts = Counter(item['db_id'] for item in all_data)
        stats["database_distribution"] = dict(db_counts.most_common(10))
        
        # SQL complexity analysis
        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", 
                       "HAVING", "UNION", "INTERSECT", "EXCEPT"]
        
        keyword_counts = defaultdict(int)
        for item in all_data:
            sql_upper = item['sql'].upper()
            for keyword in sql_keywords:
                if keyword in sql_upper:
                    keyword_counts[keyword] += 1
        
        stats["sql_keyword_frequency"] = dict(keyword_counts)
        
        # Question length distribution
        question_lengths = [len(item['question'].split()) for item in all_data]
        stats["question_length_stats"] = {
            "min": min(question_lengths),
            "max": max(question_lengths),
            "mean": np.mean(question_lengths),
            "median": np.median(question_lengths),
            "std": np.std(question_lengths)
        }
        
        return stats
    
    def save_split(self, split: DatasetSplit, output_dir: str, split_name: str = "custom"):
        """Save a dataset split to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save train
        with open(output_path / f"{split_name}_train.json", 'w') as f:
            json.dump(split.train, f, indent=2)
        
        # Save dev
        with open(output_path / f"{split_name}_dev.json", 'w') as f:
            json.dump(split.dev, f, indent=2)
        
        # Save test if exists
        if split.test:
            with open(output_path / f"{split_name}_test.json", 'w') as f:
                json.dump(split.test, f, indent=2)
        
        # Save metadata
        metadata = {
            "split_name": split_name,
            "train_size": len(split.train),
            "dev_size": len(split.dev),
            "test_size": len(split.test) if split.test else 0,
            "total_size": len(split)
        }
        
        with open(output_path / f"{split_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Split saved to {output_path}")
