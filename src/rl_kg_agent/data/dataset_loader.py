"""HuggingFace dataset integration for loading QA pairs for training."""

from typing import Dict, List, Any, Optional, Tuple, Iterator
import logging
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re


logger = logging.getLogger(__name__)


class QADatasetLoader:
    """Loads and processes QA datasets from HuggingFace and other sources."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize dataset loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.loaded_datasets: Dict[str, Dataset] = {}
        self.entity_extractor = EntityExtractor()

    def load_squad_dataset(self, version: str = "2.0", split: str = "train") -> Dataset:
        """Load SQuAD dataset for question answering.

        Args:
            version: SQuAD version ("1.1" or "2.0")
            split: Dataset split ("train", "validation")

        Returns:
            Loaded dataset
        """
        dataset_name = f"squad_v{version.replace('.', '_')}"

        if dataset_name not in self.loaded_datasets:
            try:
                logger.info(f"Loading SQuAD v{version} dataset")

                if version == "2.0":
                    dataset = load_dataset("squad_v2", split=split, cache_dir=self.cache_dir)
                else:
                    dataset = load_dataset("squad", split=split, cache_dir=self.cache_dir)

                # Process the dataset
                processed_dataset = self._process_squad_dataset(dataset)
                self.loaded_datasets[dataset_name] = processed_dataset

                logger.info(f"Loaded {len(processed_dataset)} examples from SQuAD v{version}")

            except Exception as e:
                logger.error(f"Failed to load SQuAD dataset: {e}")
                raise

        return self.loaded_datasets[dataset_name]

    def load_natural_questions(self, split: str = "train", sample_size: Optional[int] = None) -> Dataset:
        """Load Natural Questions dataset.

        Args:
            split: Dataset split
            sample_size: Optional sample size to limit dataset

        Returns:
            Loaded dataset
        """
        dataset_name = "natural_questions"

        if dataset_name not in self.loaded_datasets:
            try:
                logger.info("Loading Natural Questions dataset")

                # Load the dataset (this is a large dataset, so we might want to stream)
                dataset = load_dataset(
                    "natural_questions",
                    split=split,
                    cache_dir=self.cache_dir,
                    streaming=True if sample_size and sample_size < 10000 else False
                )

                if sample_size:
                    if hasattr(dataset, 'take'):
                        dataset = dataset.take(sample_size)
                    else:
                        dataset = dataset.select(range(min(sample_size, len(dataset))))

                # Process the dataset
                processed_dataset = self._process_natural_questions_dataset(dataset)
                self.loaded_datasets[dataset_name] = processed_dataset

                logger.info(f"Loaded {len(processed_dataset)} examples from Natural Questions")

            except Exception as e:
                logger.error(f"Failed to load Natural Questions dataset: {e}")
                # Fallback to creating a minimal example dataset
                processed_dataset = self._create_fallback_dataset()
                self.loaded_datasets[dataset_name] = processed_dataset

        return self.loaded_datasets[dataset_name]

    def load_ms_marco(self, split: str = "train", sample_size: Optional[int] = None) -> Dataset:
        """Load MS MARCO dataset for question answering.

        Args:
            split: Dataset split
            sample_size: Optional sample size

        Returns:
            Loaded dataset
        """
        dataset_name = "ms_marco"

        if dataset_name not in self.loaded_datasets:
            try:
                logger.info("Loading MS MARCO dataset")

                dataset = load_dataset(
                    "ms_marco",
                    "v1.1",
                    split=split,
                    cache_dir=self.cache_dir
                )

                if sample_size:
                    dataset = dataset.select(range(min(sample_size, len(dataset))))

                # Process the dataset
                processed_dataset = self._process_ms_marco_dataset(dataset)
                self.loaded_datasets[dataset_name] = processed_dataset

                logger.info(f"Loaded {len(processed_dataset)} examples from MS MARCO")

            except Exception as e:
                logger.error(f"Failed to load MS MARCO dataset: {e}")
                processed_dataset = self._create_fallback_dataset()
                self.loaded_datasets[dataset_name] = processed_dataset

        return self.loaded_datasets[dataset_name]

    def load_custom_dataset(self, file_path: str, format: str = "json") -> Dataset:
        """Load custom QA dataset from file.

        Args:
            file_path: Path to the dataset file
            format: File format ("json", "csv", "jsonl")

        Returns:
            Loaded dataset
        """
        try:
            logger.info(f"Loading custom dataset from {file_path}")

            if format == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif format == "jsonl":
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            elif format == "csv":
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Convert to HuggingFace dataset
            dataset = Dataset.from_list(data)

            # Ensure standard format
            dataset = self._standardize_dataset_format(dataset)

            return dataset

        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
            raise

    def _process_squad_dataset(self, dataset: Dataset) -> Dataset:
        """Process SQuAD dataset to standard format."""
        def process_example(example):
            # Extract entities from question
            entities = self.entity_extractor.extract_entities(example["question"])

            # Handle answers
            if example["answers"]["text"]:
                answer = example["answers"]["text"][0]
                has_answer = True
            else:
                answer = ""
                has_answer = False

            return {
                "question": example["question"],
                "answer": answer,
                "context": example["context"],
                "entities": entities,
                "has_answer": has_answer,
                "dataset_source": "squad",
                "question_type": self._classify_question_type(example["question"])
            }

        return dataset.map(process_example)

    def _process_natural_questions_dataset(self, dataset) -> Dataset:
        """Process Natural Questions dataset to standard format."""
        processed_examples = []

        try:
            for example in dataset:
                if isinstance(example, dict):
                    question = example.get("question", {}).get("text", "")
                    if not question:
                        continue

                    # Extract short answer if available
                    annotations = example.get("annotations", [])
                    answer = ""
                    has_answer = False

                    if annotations and len(annotations) > 0:
                        short_answers = annotations[0].get("short_answers", [])
                        if short_answers:
                            # Take first short answer
                            start_token = short_answers[0].get("start_token", 0)
                            end_token = short_answers[0].get("end_token", 0)

                            document = example.get("document", {})
                            tokens = document.get("tokens", [])

                            if start_token < len(tokens) and end_token <= len(tokens):
                                answer_tokens = tokens[start_token:end_token]
                                answer = " ".join([token.get("token", "") for token in answer_tokens])
                                has_answer = True

                    entities = self.entity_extractor.extract_entities(question)

                    processed_examples.append({
                        "question": question,
                        "answer": answer,
                        "context": "",  # Natural Questions doesn't have explicit context
                        "entities": entities,
                        "has_answer": has_answer,
                        "dataset_source": "natural_questions",
                        "question_type": self._classify_question_type(question)
                    })

                    if len(processed_examples) >= 1000:  # Limit to prevent memory issues
                        break

        except Exception as e:
            logger.warning(f"Error processing Natural Questions example: {e}")

        if not processed_examples:
            return self._create_fallback_dataset()

        return Dataset.from_list(processed_examples)

    def _process_ms_marco_dataset(self, dataset: Dataset) -> Dataset:
        """Process MS MARCO dataset to standard format."""
        def process_example(example):
            question = example.get("query", "")
            passages = example.get("passages", [])

            # Get best passage as context
            context = ""
            answer = ""

            if passages:
                # Find passage marked as selected or take first one
                for passage in passages:
                    if passage.get("is_selected", False):
                        context = passage.get("passage_text", "")
                        break

                if not context and passages:
                    context = passages[0].get("passage_text", "")

            # Try to extract answer from context
            if context and question:
                answer = self._extract_answer_from_context(question, context)

            entities = self.entity_extractor.extract_entities(question)

            return {
                "question": question,
                "answer": answer,
                "context": context,
                "entities": entities,
                "has_answer": bool(answer),
                "dataset_source": "ms_marco",
                "question_type": self._classify_question_type(question)
            }

        return dataset.map(process_example)

    def _standardize_dataset_format(self, dataset: Dataset) -> Dataset:
        """Ensure dataset has standard format."""
        required_columns = ["question", "answer", "context", "entities", "has_answer"]

        def standardize_example(example):
            standardized = {}

            # Map common column names
            question_cols = ["question", "query", "input", "text"]
            answer_cols = ["answer", "target", "output", "response"]
            context_cols = ["context", "passage", "document", "background"]

            # Find question
            for col in question_cols:
                if col in example and example[col]:
                    standardized["question"] = str(example[col])
                    break
            else:
                standardized["question"] = ""

            # Find answer
            for col in answer_cols:
                if col in example and example[col]:
                    standardized["answer"] = str(example[col])
                    break
            else:
                standardized["answer"] = ""

            # Find context
            for col in context_cols:
                if col in example and example[col]:
                    standardized["context"] = str(example[col])
                    break
            else:
                standardized["context"] = ""

            # Extract entities if not present
            if "entities" not in example:
                standardized["entities"] = self.entity_extractor.extract_entities(
                    standardized["question"]
                )
            else:
                standardized["entities"] = example["entities"]

            # Set has_answer
            standardized["has_answer"] = bool(standardized["answer"])

            # Set defaults for other fields
            standardized["dataset_source"] = example.get("dataset_source", "custom")
            standardized["question_type"] = example.get(
                "question_type",
                self._classify_question_type(standardized["question"])
            )

            return standardized

        return dataset.map(standardize_example)

    def _create_fallback_dataset(self) -> Dataset:
        """Create a fallback dataset with sample questions."""
        sample_data = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "context": "France is a country in Europe. Its capital city is Paris.",
                "entities": ["France", "Paris"],
                "has_answer": True,
                "dataset_source": "fallback",
                "question_type": "factual"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare",
                "context": "Romeo and Juliet is a tragedy written by William Shakespeare.",
                "entities": ["Romeo", "Juliet", "William Shakespeare"],
                "has_answer": True,
                "dataset_source": "fallback",
                "question_type": "factual"
            },
            {
                "question": "How do photosynthesis work?",
                "answer": "Photosynthesis converts light energy into chemical energy in plants.",
                "context": "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                "entities": ["photosynthesis", "plants"],
                "has_answer": True,
                "dataset_source": "fallback",
                "question_type": "explanation"
            }
        ]

        logger.info("Created fallback dataset with sample questions")
        return Dataset.from_list(sample_data)

    def _classify_question_type(self, question: str) -> str:
        """Classify question type based on question words."""
        question_lower = question.lower().strip()

        if question_lower.startswith(("what", "which")):
            return "factual"
        elif question_lower.startswith("who"):
            return "person"
        elif question_lower.startswith("when"):
            return "temporal"
        elif question_lower.startswith("where"):
            return "location"
        elif question_lower.startswith("how"):
            return "explanation"
        elif question_lower.startswith("why"):
            return "causal"
        elif question_lower.startswith(("is", "are", "does", "did", "can", "will")):
            return "boolean"
        else:
            return "other"

    def _extract_answer_from_context(self, question: str, context: str) -> str:
        """Simple answer extraction from context (placeholder)."""
        # This is a simplified approach - could be enhanced with NER or more sophisticated methods
        question_words = set(question.lower().split())
        context_sentences = context.split(".")

        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))

            if overlap >= 2:  # Simple heuristic
                return sentence.strip()

        return ""

    def create_training_batch(self, datasets: List[str], batch_size: int = 32,
                            question_types: Optional[List[str]] = None) -> Iterator[List[Dict[str, Any]]]:
        """Create training batches from loaded datasets.

        Args:
            datasets: List of dataset names to use
            batch_size: Size of each batch
            question_types: Filter by question types

        Yields:
            Batches of training examples
        """
        # Combine all requested datasets
        combined_examples = []

        for dataset_name in datasets:
            if dataset_name in self.loaded_datasets:
                dataset = self.loaded_datasets[dataset_name]

                for example in dataset:
                    if question_types and example["question_type"] not in question_types:
                        continue
                    combined_examples.append(example)

        # Shuffle examples
        np.random.shuffle(combined_examples)

        # Yield batches
        for i in range(0, len(combined_examples), batch_size):
            batch = combined_examples[i:i + batch_size]
            yield batch

    def get_dataset_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about loaded datasets.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {}

        for name, dataset in self.loaded_datasets.items():
            question_types = {}
            has_answer_count = 0

            for example in dataset:
                q_type = example.get("question_type", "unknown")
                question_types[q_type] = question_types.get(q_type, 0) + 1

                if example.get("has_answer", False):
                    has_answer_count += 1

            stats[name] = {
                "total_examples": len(dataset),
                "question_types": question_types,
                "has_answer_ratio": has_answer_count / len(dataset) if dataset else 0,
                "columns": list(dataset.column_names) if dataset else []
            }

        return stats


class EntityExtractor:
    """Simple entity extractor for question analysis."""

    def __init__(self):
        """Initialize entity extractor."""
        # Simple patterns for entity detection (could be enhanced with NER models)
        self.patterns = {
            "person": r"\\b[A-Z][a-z]+\\s[A-Z][a-z]+\\b",
            "location": r"\\b[A-Z][a-z]+(?:\\s[A-Z][a-z]+)*(?:\\sCity|\\sCountry|\\sState)?\\b",
            "organization": r"\\b[A-Z][a-z]+(?:\\s[A-Z][a-z]+)*(?:\\sInc|\\sCorp|\\sLLC|\\sCompany)?\\b",
            "date": r"\\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\\s\\d{1,2},?\\s\\d{4}\\b|\\b\\d{4}\\b"
        }

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using simple patterns.

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        entities = []

        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            entities.extend(matches)

        # Remove duplicates and filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        entities = list(set([
            entity for entity in entities
            if entity.lower() not in stop_words and len(entity) > 2
        ]))

        return entities[:10]  # Limit to top 10 entities