"""
Manually download the Sentence Transformer model.
Run this ONCE before running the main script.
"""
print("Downloading all-mpnet-base-v2 model...")
print("This is a one-time 400MB download.")
print()

try:
    from sentence_transformers import SentenceTransformer
    import time

    start = time.time()
    print("[1/2] Importing sentence_transformers... OK")

    print("[2/2] Downloading model (this may take 2-5 minutes depending on internet speed)...")
    model = SentenceTransformer('all-mpnet-base-v2')

    elapsed = time.time() - start
    dim = model.get_sentence_embedding_dimension()

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"Model: all-mpnet-base-v2")
    print(f"Dimensions: {dim}")
    print(f"Download time: {elapsed:.1f}s")
    print()
    print("You can now run the 270-protein conjecture script.")
    print("=" * 60)

except Exception as e:
    print()
    print("=" * 60)
    print("FAILED")
    print("=" * 60)
    print(f"Error: {e}")
    print()
    print("Possible issues:")
    print("1. No internet connection")
    print("2. Firewall blocking HuggingFace (huggingface.co)")
    print("3. Antivirus blocking downloads")
    print("4. Insufficient disk space")
    print()
    import traceback
    traceback.print_exc()
