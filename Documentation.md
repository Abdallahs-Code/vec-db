Vector Storage Layer Documentation
1. Overview
-------------
This document explains how the vector database stores, reads, and samples high-dimensional vectors efficiently using disk-based memory-mapped files. The goal is to handle datasets ranging from thousands to millions of vectors while keeping RAM usage extremely low.

The storage layer provides:
    - Fast sequential writes
    - Constant-time random access to any vector
    - Ability to sample vectors without loading the entire dataset
    - Scalable performance for large datasets
====================================================================================
2. Data Format
----------------
Each vector:
    - Has 64 dimensions
    - Is stored in float32 format
    - Uses exactly 64 × 4 = 256 bytes per vector

All vectors are written contiguously in a single binary file.

There is no metadata stored inside the file.
The system infers vector count from file size.
====================================================================================
3. File Layout

The database file (saved_db.dat) is a simple flat array of float32 values:

| vec0[0] | vec0[1] | ... | vec0[63] |
| vec1[0] | vec1[1] | ... | vec1[63] |
| vec2[0] | vec2[1] | ... | vec2[63] |
...

All vectors follow each other back-to-back.
====================================================================================
4. Offset Rule for Accessing a Row

To read vector i, the system must jump to the correct byte location inside the file.
Offset in bytes = i × DIMENSION × size_of_float32

With:
    - DIMENSION = 64
    - size_of_float32 = 4 bytes

So the rule becomes:
    offset = i × 64 × 4

Example:
Vector 0 → offset = 0
Vector 1 → offset = 1 × 64 × 4 = 256 bytes
Vector 10 → offset = 10 × 64 × 4 = 2560 bytes

This allows accessing any vector in constant time with no scanning.
====================================================================================
5. Writing Vectors to Disk

Vectors are written using numpy.memmap:
    - The file is opened in w+ mode
    - A memory-mapped array is created with shape (num_vectors, 70)
    - Data is copied directly into the mapped region
    - flush() ensures all bytes are written to disk

Advantages:
    - Handles very large datasets
    - No need to allocate giant RAM arrays
    - Sequential write pattern is extremely fast
====================================================================================
6. Reading a Single Vector

To read one vector without loading the entire database:
    1. Compute the offset using the rule above
    2. Create a small memmap of shape (1, 70)
    3. Convert it into a normal NumPy array
    4. Return the vector

This operation requires only a few kilobytes of RAM, regardless of database size.
====================================================================================
7. Reading All Vectors

A full read uses:
    vectors = np.memmap(path, dtype=float32, mode='r', shape=(num_records, 70))
Then it converts the memmap to a NumPy array.

Note: this loads the entire database into RAM.
It should be used only in evaluation or debugging.
====================================================================================
8. Inserting New Vectors

To insert new data:
    1. Determine the old number of records
    2. Expand the memory-mapped file to fit new rows
    3. Append the new vectors starting at the new offset
    4. Flush the file

This keeps the file layout consistent and append-only.
====================================================================================
9. Dynamic Clustering Parameters

The system automatically configures clustering parameters based on database size.

Sample Size:
    - 5% of database size
Number of Clusters:
    - Nearest power of 2 of sqrt(db_size)
    - Example: 1M vectors → 1024 clusters
Batch Size:
    - Fixed at 4096 vectors per batch
    - Optimized for MiniBatch K-Means
Max Iterations:
    - Fixed at 200 iterations
    - Sufficient for convergence across all database sizes
====================================================================================
10. Sampling Vectors Efficiently

Sampling must avoid loading all vectors into memory.

The storage layer uses this strategy:
    1. Determine total number of records
    2. Randomly select indices
    3. For each index, call get_one_row()
    4. Collect the sampled vectors into a small NumPy array

This method scales well even for millions of vectors.
====================================================================================
11. Training Centroids

Trains MiniBatchKMeans on sampled vectors to generate cluster centroids.

Input:
    - sampled_vectors: NumPy array of shape (sample_size, 64)
Process:
    - Uses MiniBatchKMeans with parameters from compute_clustering_parameters()
    - random_state=42 ensures reproducible results
    - Fits the model on sampled vectors
Output:
    - Returns cluster centroids as float32 array
    - Shape: (n_clusters, 64)
====================================================================================
12. Saving Centroids

Saves cluster centroids to disk using the same binary format as the main database.

Input:
    - centroids: NumPy array of shape (n_clusters, 64), dtype=float32
Process:
    - Creates memory-mapped file at index_path
    - Writes centroids contiguously in float32 format
    - Flushes to ensure data is written to disk
File Format:
    - Same layout as main database (no headers or metadata)
    - Each centroid is 64 × 4 = 256 bytes
    - Total file size = n_clusters × 256 bytes
====================================================================================
13. Summary
To sum up:
- Vectors are stored contiguously in float32 format
- Every vector is exactly 256 bytes
- Single-vector access uses the offset rule
- Memory mapping enables scalable performance
- Sampling and insertion operations work without loading the whole dataset
- This layer provides a clean, efficient foundation for indexing and search algorithms