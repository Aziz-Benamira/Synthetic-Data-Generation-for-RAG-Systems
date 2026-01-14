"""
Example Usage of Synthesize-on-Graph (SoG) Pipeline

This script demonstrates how to use the SoG framework to generate
synthetic data from a corpus of documents.
"""

import json
from pathlib import Path

from sog_pipeline import (
    SynthesizeOnGraph,
    SoGConfig,
    Document,
    load_documents_from_json,
    load_documents_from_jsonl
)
from entity_extraction import SimpleEntityExtractor, analyze_entity_distribution
from text_processing import split_into_paragraphs, clean_text
from generation_strategies import MockLLMInterface


def create_sample_documents():
    """
    Create sample documents for demonstration.
    
    In production, replace this with your actual document loading logic.
    """
    documents = [
        Document(
            doc_id=0,
            title="Introduction to Machine Learning",
            paragraphs=[
                "Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from data. These algorithms improve their performance over time without being explicitly programmed.",
                "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model makes predictions based on input-output pairs provided during training.",
                "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. This approach has achieved remarkable success in computer vision and natural language processing tasks."
            ],
            entities_per_paragraph=[
                ["Machine Learning", "Artificial Intelligence"],
                ["Supervised Learning", "Algorithm", "Training Data"],
                ["Deep Learning", "Neural Networks", "Computer Vision", "Natural Language Processing"]
            ]
        ),
        Document(
            doc_id=1,
            title="Neural Networks and Deep Learning",
            paragraphs=[
                "Neural networks are computational models inspired by biological neural networks in the brain. They consist of interconnected nodes organized in layers that process information.",
                "Convolutional neural networks excel at processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies of features.",
                "Recurrent neural networks are designed for sequential data processing. They maintain internal memory states that allow them to process sequences of variable length."
            ],
            entities_per_paragraph=[
                ["Neural Networks", "Computational Models"],
                ["Convolutional Neural Networks", "Images", "Spatial Hierarchies"],
                ["Recurrent Neural Networks", "Sequential Data", "Memory States"]
            ]
        ),
        Document(
            doc_id=2,
            title="Natural Language Processing Applications",
            paragraphs=[
                "Natural language processing enables computers to understand and generate human language. Modern NLP systems use deep learning models trained on large text corpora.",
                "Machine translation automatically converts text from one language to another. Neural machine translation models have significantly improved translation quality in recent years.",
                "Question answering systems can understand natural language queries and retrieve relevant information. These systems combine information retrieval with natural language understanding."
            ],
            entities_per_paragraph=[
                ["Natural Language Processing", "Deep Learning", "Text Corpora"],
                ["Machine Translation", "Neural Machine Translation"],
                ["Question Answering", "Information Retrieval", "Natural Language Understanding"]
            ]
        ),
        Document(
            doc_id=3,
            title="Computer Vision and Image Recognition",
            paragraphs=[
                "Computer vision focuses on enabling machines to interpret and understand visual information from the world. Deep learning has revolutionized this field through convolutional neural networks.",
                "Object detection identifies and localizes multiple objects within images. Modern detectors like YOLO and Faster R-CNN achieve real-time performance with high accuracy.",
                "Image segmentation partitions images into meaningful regions. Semantic segmentation assigns class labels to pixels while instance segmentation distinguishes individual objects."
            ],
            entities_per_paragraph=[
                ["Computer Vision", "Visual Information", "Deep Learning", "Convolutional Neural Networks"],
                ["Object Detection", "YOLO", "Faster R-CNN"],
                ["Image Segmentation", "Semantic Segmentation", "Instance Segmentation"]
            ]
        )
    ]
    
    return documents


def example_basic_usage():
    """
    Example 1: Basic usage with default configuration.
    """
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Initialize SoG pipeline with default configuration
    config = SoGConfig(
        max_depth=2,
        num_samples=10,
        output_path="output/synthetic_data_basic.jsonl",
        verbose=True
    )
    
    sog = SynthesizeOnGraph(config=config)
    
    # Run pipeline
    samples = sog.run_pipeline(documents, num_samples=10)
    
    # Print some samples
    print(f"\nGenerated {len(samples)} samples")
    if samples:
        print("\nFirst sample:")
        sample = samples[0]
        print(f"Question: {sample.question}")
        print(f"Answer: {sample.answer}")
        print(f"Strategy: {sample.strategy_used.value}")
        print(f"Reasoning Type: {sample.reasoning_type}")
    
    # Print statistics
    stats = sog.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))


def example_custom_configuration():
    """
    Example 2: Custom configuration with different parameters.
    """
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)
    
    documents = create_sample_documents()
    
    # Custom configuration
    config = SoGConfig(
        max_depth=3,  # Deeper paths
        top_k_neighbors=3,  # Fewer neighbors at each step
        long_tail_boost=2.0,  # Stronger boost for long-tail entities
        num_samples=15,
        sparse_threshold=2,
        output_path="output/synthetic_data_custom.jsonl",
        random_seed=123,
        verbose=True
    )
    
    sog = SynthesizeOnGraph(config=config)
    samples = sog.run_pipeline(documents, num_samples=15)
    
    print(f"\nGenerated {len(samples)} samples with custom configuration")


def example_from_text_files():
    """
    Example 3: Loading documents from text files with entity extraction.
    """
    print("\n" + "=" * 80)
    print("Example 3: Processing Text Files")
    print("=" * 80)
    
    # Simulate loading from text files
    text_files = {
        "doc1.txt": """
        Retrieval-Augmented Generation combines the power of large language models
        with external knowledge retrieval. This approach enables LLMs to access
        information beyond their training data.
        
        The retrieval component searches through a knowledge base to find relevant
        documents. These documents are then provided as context to the language model
        for generation.
        """,
        "doc2.txt": """
        Knowledge graphs represent structured information as nodes and edges.
        Entities are represented as nodes while relationships between entities
        form the edges of the graph.
        
        Graph-based retrieval can leverage these structured relationships to find
        more relevant information. This approach is particularly effective for
        multi-hop reasoning tasks.
        """
    }
    
    # Process texts and extract entities
    entity_extractor = SimpleEntityExtractor()
    documents = []
    
    for doc_id, (filename, text) in enumerate(text_files.items()):
        # Split into paragraphs
        paragraphs = split_into_paragraphs(text, min_length=30)
        
        # Extract entities from each paragraph
        entities_per_paragraph = []
        for para in paragraphs:
            entities = entity_extractor.extract_entities(para)
            entities_per_paragraph.append(entities)
        
        # Create document
        doc = Document(
            doc_id=doc_id,
            title=filename,
            paragraphs=paragraphs,
            entities_per_paragraph=entities_per_paragraph
        )
        documents.append(doc)
    
    # Analyze entity distribution
    all_entities = [e for doc in documents for e in doc.entities_per_paragraph]
    analysis = analyze_entity_distribution(all_entities)
    print("\nEntity Distribution Analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Run SoG pipeline
    config = SoGConfig(
        max_depth=2,
        num_samples=5,
        output_path="output/synthetic_data_from_texts.jsonl",
        verbose=True
    )
    
    sog = SynthesizeOnGraph(config=config)
    samples = sog.run_pipeline(documents, num_samples=5)
    
    print(f"\nGenerated {len(samples)} samples from text files")


def example_incremental_processing():
    """
    Example 4: Incremental document addition and generation.
    """
    print("\n" + "=" * 80)
    print("Example 4: Incremental Processing")
    print("=" * 80)
    
    config = SoGConfig(
        max_depth=2,
        num_samples=5,
        verbose=True
    )
    
    sog = SynthesizeOnGraph(config=config)
    
    # Add documents in batches
    documents_batch1 = create_sample_documents()[:2]
    documents_batch2 = create_sample_documents()[2:]
    
    print("Adding first batch of documents...")
    sog.add_documents(documents_batch1)
    
    print("\nGenerating samples from first batch...")
    samples1 = sog.generate_synthetic_data(num_samples=3)
    print(f"Generated {len(samples1)} samples")
    
    print("\nAdding second batch of documents...")
    sog.add_documents(documents_batch2)
    
    print("\nGenerating samples from complete corpus...")
    samples2 = sog.generate_synthetic_data(num_samples=5)
    print(f"Generated {len(samples2)} samples")
    
    # Save all samples
    all_samples = samples1 + samples2
    sog.save_samples(all_samples, "output/synthetic_data_incremental.jsonl")


def example_analysis_only():
    """
    Example 5: Analyze corpus without generation.
    """
    print("\n" + "=" * 80)
    print("Example 5: Corpus Analysis")
    print("=" * 80)
    
    documents = create_sample_documents()
    
    # Build context graph
    sog = SynthesizeOnGraph()
    sog.add_documents(documents)
    
    # Get detailed statistics
    stats = sog.get_statistics()
    
    print("\nDetailed Graph Statistics:")
    print(json.dumps(stats['graph'], indent=2))
    
    # Analyze entity distribution
    all_entities = [e for doc in documents for e in doc.entities_per_paragraph]
    entity_analysis = analyze_entity_distribution(all_entities)
    
    print("\nEntity Distribution:")
    print(json.dumps(entity_analysis, indent=2))
    
    # Get long-tail entities
    long_tail = sog.context_graph.get_long_tail_entities(percentile=0.25)
    print(f"\nLong-tail entities (bottom 25%): {len(long_tail)}")
    print("Examples:", list(long_tail)[:10])


def main():
    """
    Run all examples.
    """
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    print("Synthesize-on-Graph (SoG) Framework Examples")
    print("=" * 80)
    
    # Run examples
    example_basic_usage()
    example_custom_configuration()
    example_from_text_files()
    example_incremental_processing()
    example_analysis_only()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("Check the 'output' directory for generated files.")
    print("=" * 80)


if __name__ == "__main__":
    main()
