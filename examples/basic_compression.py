"""
Basic Compression Example
=========================

Demonstrates basic usage of sageRefiner for context compression.
"""

from sageRefiner import LongRefiner, RefinerConfig


def main():
    # Sample documents
    documents = [
        "Regular exercise provides numerous health benefits. It strengthens the cardiovascular system, "
        "improves muscle tone, and enhances overall physical fitness. Studies show that 30 minutes of "
        "moderate exercise daily can significantly reduce the risk of chronic diseases.",
        "Physical activity is crucial for mental wellbeing. Exercise releases endorphins, which are "
        "natural mood elevators. It also reduces stress hormones like cortisol and adrenaline. Many "
        "people report improved sleep quality and reduced anxiety after regular physical activity.",
        "Weight management is another key benefit of exercise. Combined with proper nutrition, regular "
        "physical activity helps maintain a healthy body weight by burning calories and building lean "
        "muscle mass. This metabolic boost continues even after the workout is complete.",
        "Social benefits of exercise shouldn't be overlooked. Group fitness classes, team sports, and "
        "workout partners can provide motivation and accountability. These social connections often "
        "extend beyond the gym, creating supportive communities.",
    ]

    query = "What are the main benefits of regular exercise?"

    print("=" * 80)
    print("sageRefiner - Basic Compression Example")
    print("=" * 80)
    print(f"\nQuery: {query}")
    print(f"Documents: {len(documents)} items")
    print(f"Original total length: {sum(len(d) for d in documents)} characters\n")

    # Configure refiner
    config = RefinerConfig(
        algorithm="long_refiner",
        budget=512,  # Target token count
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",  # Use small model for demo
        compression_ratio=0.5,
        device="cuda",  # Change to "cpu" if no GPU available
    )

    print("Configuration:")
    print(f"  Algorithm: {config.algorithm.value}")
    print(f"  Budget: {config.budget} tokens")
    print(f"  Model: {config.base_model_path}")
    print(f"  Compression Ratio: {config.compression_ratio}")
    print()

    # Initialize refiner
    print("Initializing refiner...")
    refiner = LongRefiner(config.to_dict())
    refiner.initialize()
    print("Refiner ready.\n")

    # Perform compression
    print("Compressing documents...")
    result = refiner.refine(query, documents, budget=config.budget)

    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Original tokens:   {result.metrics.original_tokens}")
    print(f"Compressed tokens: {result.metrics.refined_tokens}")
    print(f"Compression ratio: {result.metrics.compression_rate:.2f}x")
    print(f"Processing time:   {result.metrics.refine_time:.2f}s")
    print()

    print("Compressed Content:")
    print("-" * 80)
    for i, content in enumerate(result.refined_content, 1):
        print(f"\n[Document {i}]")
        print(content)

    print("\n" + "=" * 80)
    print("Compression complete.")
    print("=" * 80)

    # Cleanup
    refiner.shutdown()


if __name__ == "__main__":
    main()
