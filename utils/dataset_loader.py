"""
Main dataset loader interface for Text2SQL experiments.
Provides easy access to different dataset splits and configurations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from utils.others.dataset_utils import SpiderDatasetSplitter, DatasetSplit

class SpiderDatasetLoader:
    """
    Main interface for loading Spider dataset with different configurations.
    Handles caching and provides convenient methods for experiments.
    """
    
    def __init__(self, data_dir: str = "./data/spider_dataset"):
        self.data_dir = Path(data_dir)
        self.splitter = SpiderDatasetSplitter(data_dir)
        self.cache = {}
        
    def load_original(self) -> DatasetSplit:
        """Load the original Spider train/dev split."""
        if "original" not in self.cache:
            train_data, dev_data = self.splitter.load_raw_data()
            self.cache["original"] = DatasetSplit(
                train=train_data,
                dev=dev_data,
                test=None
            )
        return self.cache["original"]
    
    def load_random_split(self, 
                         train_ratio: float = 0.7,
                         dev_ratio: float = 0.15, 
                         test_ratio: float = 0.15,
                         seed: int = 42) -> DatasetSplit:
        """Load a random split of the data."""
        cache_key = f"random_{train_ratio}_{dev_ratio}_{test_ratio}_{seed}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self.splitter.random_split(
                train_ratio, dev_ratio, test_ratio, seed
            )
        return self.cache[cache_key]
    
    def load_few_shot(self, 
                     n_shot: int = 5, 
                     n_eval: int = 100,
                     strategy: str = "random") -> DatasetSplit:
        """Load a few-shot configuration."""
        cache_key = f"few_shot_{n_shot}_{n_eval}_{strategy}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self.splitter.few_shot_split(n_shot, n_eval, strategy)
        return self.cache[cache_key]
    
    def load_difficulty_split(self, difficulty: str) -> DatasetSplit:
        """Load examples of specific difficulty level."""
        if "difficulty_splits" not in self.cache:
            self.cache["difficulty_splits"] = self.splitter.difficulty_split()
        
        if difficulty not in self.cache["difficulty_splits"]:
            raise ValueError(f"Unknown difficulty: {difficulty}. Available: {list(self.cache['difficulty_splits'].keys())}")
        
        return self.cache["difficulty_splits"][difficulty]
    
    def get_examples_by_database(self, db_id: str, split: str = "original") -> List[Dict]:
        """Get all examples for a specific database."""
        dataset = self.load_original() if split == "original" else self.load_random_split()
        
        examples = []
        for data_split in [dataset.train, dataset.dev]:
            if dataset.test:
                data_split = [dataset.train, dataset.dev, dataset.test]
            else:
                data_split = [dataset.train, dataset.dev]
            
            for split_data in data_split:
                for item in split_data:
                    if item['db_id'] == db_id:
                        examples.append(item)
        
        return examples
    
    def get_sample_for_prompt(self, 
                             n_examples: int = 3,
                             exclude_db: Optional[str] = None,
                             difficulty: Optional[str] = None) -> List[Dict]:
        """
        Get sample examples for prompt engineering.
        
        Args:
            n_examples: Number of examples to return
            exclude_db: Database to exclude (for avoiding data leakage)
            difficulty: Specific difficulty level
        """
        if difficulty:
            dataset = self.load_difficulty_split(difficulty)
            candidates = dataset.train
        else:
            dataset = self.load_original()
            candidates = dataset.train
        
        # Filter out excluded database
        if exclude_db:
            candidates = [item for item in candidates if item['db_id'] != exclude_db]
        
        # Sample examples
        import random
        if len(candidates) <= n_examples:
            return candidates
        
        return random.sample(candidates, n_examples)
    
    def get_evaluation_set(self, 
                          size: Optional[int] = None,
                          databases: Optional[List[str]] = None,
                          difficulty: Optional[str] = None) -> List[Dict]:
        """
        Get a configured evaluation set.
        
        Args:
            size: Maximum number of examples
            databases: Specific databases to include
            difficulty: Specific difficulty level
        """
        if difficulty:
            dataset = self.load_difficulty_split(difficulty)
            candidates = dataset.dev
        else:
            dataset = self.load_original()
            candidates = dataset.dev
        
        # Filter by databases
        if databases:
            candidates = [item for item in candidates if item['db_id'] in databases]
        
        # Limit size
        if size and len(candidates) > size:
            import random
            candidates = random.sample(candidates, size)
        
        return candidates
    
    def create_experiment_config(self, 
                               experiment_name: str,
                               train_config: Dict,
                               eval_config: Dict) -> Dict:
        """
        Create a complete experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            train_config: Configuration for training/few-shot examples
            eval_config: Configuration for evaluation set
        """
        config = {
            "experiment_name": experiment_name,
            "train_config": train_config,
            "eval_config": eval_config,
            "datasets": {}
        }
        
        # Load training data based on config
        if train_config["type"] == "few_shot":
            train_data = self.load_few_shot(
                n_shot=train_config.get("n_shot", 5),
                n_eval=0,  # We'll use eval_config for evaluation
                strategy=train_config.get("strategy", "random")
            )
            config["datasets"]["train"] = train_data.train
        elif train_config["type"] == "original":
            dataset = self.load_original()
            config["datasets"]["train"] = dataset.train
        
        # Load evaluation data based on config
        eval_data = self.get_evaluation_set(
            size=eval_config.get("size"),
            databases=eval_config.get("databases"),
            difficulty=eval_config.get("difficulty")
        )
        config["datasets"]["eval"] = eval_data
        
        return config

# Convenience functions for common use cases
def quick_load(split_type: str = "original", **kwargs) -> DatasetSplit:
    """Quick load function for common splits."""
    loader = SpiderDatasetLoader()
    
    if split_type == "original":
        return loader.load_original()
    elif split_type == "random":
        return loader.load_random_split(**kwargs)
    elif split_type == "few_shot":
        return loader.load_few_shot(**kwargs)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

def create_few_shot_experiment(n_shot: int = 5, eval_size: int = 100) -> Dict:
    """Create a standard few-shot experiment setup."""
    loader = SpiderDatasetLoader()
    
    return loader.create_experiment_config(
        experiment_name=f"few_shot_{n_shot}",
        train_config={
            "type": "few_shot",
            "n_shot": n_shot,
            "strategy": "diverse"
        },
        eval_config={
            "size": eval_size
        }
    )

def create_cross_domain_experiment(target_domains: List[str]) -> Dict:
    """Create a cross-domain evaluation experiment."""
    loader = SpiderDatasetLoader()
    
    # Get databases for target domains (simplified)
    domain_db_map = {
        "music": ["concert_singer", "music_1", "music_2"],
        "sports": ["sports_1", "game_1"],
        "academic": ["college_1", "school_1", "student_1"],
        # Add more mappings as needed
    }
    
    target_databases = []
    for domain in target_domains:
        if domain in domain_db_map:
            target_databases.extend(domain_db_map[domain])
    
    return loader.create_experiment_config(
        experiment_name=f"cross_domain_{'_'.join(target_domains)}",
        train_config={
            "type": "original"
        },
        eval_config={
            "databases": target_databases
        }
    )