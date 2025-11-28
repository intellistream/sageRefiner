"""
Algorithm Comparison Example
============================

Compare different compression algorithms on the same dataset.
"""

import time

from sageRefiner import LongRefinerCompressor, REFORMCompressor, RefinerConfig


def generate_sample_docs():
    """Generate sample documents for testing."""
    return [
        "Artificial intelligence has transformed modern technology. Machine learning algorithms "
        "now power everything from recommendation systems to autonomous vehicles. The field "
        "continues to evolve rapidly with new breakthroughs announced regularly.",
        "Natural language processing enables computers to understand human language. Applications "
        "include chatbots, translation services, and sentiment analysis. Recent advances in "
        "transformer models have dramatically improved NLP capabilities.",
        "Computer vision allows machines to interpret visual information. This technology powers "
        "facial recognition, medical imaging analysis, and quality control in manufacturing. "
        "Deep learning has been particularly effective in this domain.",
        "Robotics combines AI with mechanical engineering. Modern robots can perform complex tasks "
        "in manufacturing, healthcare, and exploration. Collaborative robots now work safely "
        "alongside humans in many industries.",
    ]


def test_algorithm(name, refiner, query, documents, budget):
    """Test a specific algorithm and return metrics."""
    print(f"\nTesting {name}...")
    print("-" * 60)

    start_time = time.time()
    refiner.initialize()
    init_time = time.time() - start_time

    start_time = time.time()
    result = refiner.refine(query, documents, budget=budget)
    refine_time = time.time() - start_time

    print(f"  Initialization: {init_time:.2f}s")
    print(f"  Compression:    {refine_time:.2f}s")
    print(f"  Original:       {result.metrics.original_tokens} tokens")
    print(f"  Compressed:     {result.metrics.refined_tokens} tokens")
    print(f"  Ratio:          {result.metrics.compression_rate:.2f}x")

    refiner.shutdown()

    return {
        "name": name,
        "init_time": init_time,
        "refine_time": refine_time,
        "original_tokens": result.metrics.original_tokens,
        "refined_tokens": result.metrics.refined_tokens,
        "compression_rate": result.metrics.compression_rate,
        "output": result.refined_content,
    }


def main():
    query = "What are the key applications of AI technology?"
    documents = generate_sample_docs()
    budget = 256

    print("=" * 80)
    print("Algorithm Comparison")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Documents: {len(documents)}")
    print(f"Target budget: {budget} tokens\n")

    results = []

    # Test LongRefiner
    config_long = RefinerConfig(
        algorithm="long_refiner",
        budget=budget,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        compression_ratio=0.5,
        device="cpu",  # Use CPU for reproducibility
    )
    refiner_long = LongRefinerCompressor(config_long.to_dict())
    results.append(test_algorithm("LongRefiner", refiner_long, query, documents, budget))

    # Test Reform
    config_reform = RefinerConfig(
        algorithm="reform",
        budget=budget,
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
    refiner_reform = REFORMCompressor(config_reform.to_dict())
    results.append(test_algorithm("Reform", refiner_reform, query, documents, budget))

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Algorithm':<15} {'Init(s)':<10} {'Comp(s)':<10} {'Tokens':<10} {'Ratio':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<15} {r['init_time']:<10.2f} {r['refine_time']:<10.2f} "
            f"{r['original_tokens']}->{r['refined_tokens']:<5} {r['compression_rate']:<10.2f}x"
        )

    print("\n" + "=" * 80)
    print("Recommendations:")
    print("  - LongRefiner: Best for quality-critical applications")
    print("  - Reform:      Best for speed-critical applications")
    print("=" * 80)


if __name__ == "__main__":
    main()
