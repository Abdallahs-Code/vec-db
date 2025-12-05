from typing import Annotated
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import math
import json
import time
import gc
from pathlib import Path

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
DB_SEED_NUMBER = 42

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index", new_db = True, db_size = None) -> None:
        if not os.path.isfile(database_file_path):
            print("Database file does NOT exist.")
        self.db_path = database_file_path
        self.index_path = index_file_path
        self._build_index()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)]) -> None:
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        new_total_records = num_old_records + num_new_records
        with open(self.db_path, 'ab') as f:
            f.truncate(new_total_records * DIMENSION * ELEMENT_SIZE)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=(new_total_records, DIMENSION))
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        num_records = self._get_num_records()
        if row_num < 0 or row_num >= num_records:
            raise ValueError(f"Invalid row number: {row_num}. Must be between 0 and {num_records - 1}.")
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def compute_clustering_parameters(self):
        db_size = self._get_num_records()
        # CRITICAL: Fix n_clusters to reduce index size and improve consistency
        self.n_clusters = 1024  # Fixed for all DB sizes
        
        # Reduce sample size to save memory during training
        self.sample_size = min(int(0.03 * db_size), 100000)  # Max 100K samples, 3% of DB
        self.batch_size = 4096
        self.max_iter = 200

    def sample_for_kmeans(self, seed : int = DB_SEED_NUMBER) -> np.ndarray:
        sample_size = self.sample_size
        num_records = self._get_num_records()
        if sample_size > num_records:
            raise ValueError("Sample size cannot be greater than the number of records in the database.")
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(num_records, sample_size, replace=False)
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        samples = db_vectors[sample_indices]
        return np.array(samples, dtype=np.float32)

    def train_centroids(self, sampled_vectors: np.ndarray) -> np.ndarray:
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            random_state=42
        )
        kmeans.fit(sampled_vectors)
        return kmeans.cluster_centers_.astype(np.float32)
    
    def save_centroids(self, centroids: np.ndarray):
        if centroids.dtype != np.float32:
            centroids = centroids.astype(np.float32)
        clusters_file_path = os.path.join(self.index_path, "centroids.dat")
        with open(clusters_file_path, 'wb') as f:
            centroids.tofile(f)
        expected_size = centroids.shape[0] * centroids.shape[1] * ELEMENT_SIZE
        actual_size = os.path.getsize(clusters_file_path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"Centroid file size mismatch. Expected {expected_size} bytes, got {actual_size} bytes"
            )

    def load_centroids(self, n_clusters: int) -> np.ndarray:
        clusters_file_path = os.path.join(self.index_path, "centroids.dat")
        if not os.path.exists(clusters_file_path):
            raise FileNotFoundError(f"Centroids file not found: {clusters_file_path}")
        expected_size = n_clusters * DIMENSION * ELEMENT_SIZE
        actual_size = os.path.getsize(clusters_file_path)
        if actual_size != expected_size:
            raise ValueError(
                f"Centroids file size mismatch. Expected {expected_size} bytes "
                f"for {n_clusters} clusters, but got {actual_size} bytes. "
                f"File may be corrupted or n_clusters is incorrect."
            )
        with open(clusters_file_path, 'rb') as f:
            centroids = np.fromfile(f, dtype=np.float32, count=n_clusters * DIMENSION)
        return centroids.reshape(n_clusters, DIMENSION)

    def _build_index(self) -> None:
        self.compute_clustering_parameters()
        need_rebuild = self._should_rebuild_index()
        if need_rebuild:
            print("Building new index...")
            self._cleanup_old_index()
        else:
            print("Index is up-to-date. No rebuild needed.")
            return
        os.makedirs(self.index_path, exist_ok=True)
        centroids = self._get_or_create_centroids()
        num_records = self._get_num_records()
        bytes_per_vector = DIMENSION * ELEMENT_SIZE
        target_chunk_bytes = 16 * 1024 * 1024
        chunk_size = max(1, min(num_records, target_chunk_bytes // bytes_per_vector))
        chunk_size = min(chunk_size, 65536)
        cluster_paths = []
        for ci in range(self.n_clusters):
            cluster_file = os.path.join(self.index_path, f"cluster_{ci}.ids")
            with open(cluster_file, 'wb') as f:
                pass
            cluster_paths.append(cluster_file)
        cluster_counts = [0] * self.n_clusters
        try:
            db_vectors = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode='r',
                shape=(num_records, DIMENSION)
            )
            for chunk_start in range(0, num_records, chunk_size):
                chunk_end = min(num_records, chunk_start + chunk_size)
                chunk_vectors = np.array(db_vectors[chunk_start:chunk_end], dtype=np.float32)
                chunk_norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
                chunk_norms[chunk_norms == 0] = 1.0
                chunk_vectors_normalized = chunk_vectors / chunk_norms
                similarities = np.dot(chunk_vectors_normalized, centroids.T)
                cluster_assignments = np.argmax(similarities, axis=1)
                for cluster_id in range(self.n_clusters):
                    mask = (cluster_assignments == cluster_id)
                    local_indices = np.where(mask)[0]
                    if local_indices.size == 0:
                        continue
                    global_ids = (chunk_start + local_indices).astype(np.uint32)
                    with open(cluster_paths[cluster_id], 'ab') as f:
                        global_ids.tofile(f)
                    cluster_counts[cluster_id] += len(global_ids)
            del db_vectors
        except Exception as e:
            raise RuntimeError(f"Error during vector assignment: {e}")
        metadata = {
            "n_clusters": int(self.n_clusters),
            "num_records": int(num_records),
            "dimension": int(DIMENSION),
            "cluster_files": {
                str(i): os.path.basename(cluster_paths[i])
                for i in range(self.n_clusters)
            },
            "counts": {
                str(i): int(cluster_counts[i])
                for i in range(self.n_clusters)
            },
            "centroids_file": "centroids.dat",
            "build_timestamp": time.time()
        }
        metadata_path = os.path.join(self.index_path, "index_meta.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        for cluster_id, cluster_path in enumerate(cluster_paths):
            if not os.path.exists(cluster_path):
                raise RuntimeError(f"Cluster file {cluster_path} was not created")
        print("Index build complete.")

    def _should_rebuild_index(self) -> bool:
        centroids_path = os.path.join(self.index_path, "centroids.dat")
        if not os.path.exists(centroids_path):
            return True
        metadata_path = os.path.join(self.index_path, "index_meta.json")
        if not os.path.exists(metadata_path):
            return True
        try:
            with open(metadata_path, 'r') as f:
                old_metadata = json.load(f)
            old_n_clusters = int(old_metadata.get("n_clusters", 0))
            if old_n_clusters != self.n_clusters:
                print(f"n_clusters mismatch: old={old_n_clusters}, new={self.n_clusters}")
                return True
            old_num_records = int(old_metadata.get("num_records", 0))
            current_num_records = self._get_num_records()
            if old_num_records != current_num_records:
                print(f"num_records mismatch: old={old_num_records}, new={current_num_records}")
                return True
            expected_centroid_size = old_n_clusters * DIMENSION * ELEMENT_SIZE
            actual_centroid_size = os.path.getsize(centroids_path)
            if expected_centroid_size != actual_centroid_size:
                print(f"Centroid file corrupted: expected={expected_centroid_size}, actual={actual_centroid_size}")
                return True
            return False
        except Exception as e:
            print(f"Error reading metadata: {e}")
            return True

    def _cleanup_old_index(self):
        if os.path.isdir(self.index_path):
            try:
                import shutil
                shutil.rmtree(self.index_path)
                print(f"Removed old index directory: {self.index_path}")
            except Exception as e:
                print(f"Warning: Could not remove {self.index_path}: {e}")

    def _get_or_create_centroids(self) -> np.ndarray:
        centroids_file = os.path.join(self.index_path, "centroids.dat")
        need_training = (not os.path.exists(centroids_file) or os.path.getsize(centroids_file) == 0)
        if need_training:
            print(f"Training {self.n_clusters} centroids...")
            sampled_vectors = self.sample_for_kmeans()
            if sampled_vectors.shape[0] < self.n_clusters:
                old_n = self.n_clusters
                self.n_clusters = sampled_vectors.shape[0]
                print(f"Warning: Adjusted n_clusters from {old_n} to {self.n_clusters} due to sample size")
            centroids = self.train_centroids(sampled_vectors)
            centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroid_norms[centroid_norms == 0] = 1.0
            centroids = (centroids / centroid_norms).astype(np.float32)
            self.save_centroids(centroids)
            print(f"Centroids trained and saved to {self.index_path}")
            return centroids
        else:
            print(f"Loading existing centroids from {self.index_path}")
            centroids = self.load_centroids(self.n_clusters)
            return centroids

    def load_inverted_list(self, cluster_id: int) -> np.ndarray:
        meta_path = os.path.join(self.index_path, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        file_name = meta["cluster_files"].get(str(cluster_id))
        if file_name is None:
            return np.array([], dtype=np.uint32)
        file_path = os.path.join(self.index_path, file_name)
        if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
            return np.array([], dtype=np.uint32)
        count = os.path.getsize(file_path) // np.dtype(np.uint32).itemsize
        # Read directly into array instead of memmap to avoid page caching
        with open(file_path, 'rb') as f:
            return np.fromfile(f, dtype=np.uint32, count=count)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Ultra-low memory retrieval using minimal batch processing.
        """
        # Validate & normalize query
        qn = np.asarray(query, dtype=np.float32).reshape(-1)
        if qn.size != DIMENSION:
            raise ValueError(f"Query dimension mismatch: expected {DIMENSION}, got {qn.size}")
        q_norm = np.linalg.norm(qn)
        if q_norm > 0:
            qn /= q_norm
        
        # Load metadata
        meta_path = os.path.join(self.index_path, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Index metadata not found. Build the index first.")
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        n_clusters = int(meta["n_clusters"])
        num_records = int(meta["num_records"])
        
        # Load centroids
        centroids = self.load_centroids(n_clusters)
        sims_to_centroids = centroids.dot(qn)
        del centroids
        gc.collect()
        
        # CRITICAL: Reduce nprobe significantly to minimize candidates
        if num_records <= 1_000_000:
            nprobe = 2  # Reduced from 3
        elif num_records <= 10_000_000:
            nprobe = 3  # Reduced from 6
        else:
            nprobe = 4  # Reduced from 8
        nprobe = min(max(1, nprobe), n_clusters)
        
        # Get top-nprobe centroid indices
        top_centroid_idxs = np.argpartition(-sims_to_centroids, nprobe-1)[:nprobe]
        del sims_to_centroids
        
        # Gather candidate ids - load directly as arrays (not memmaps)
        candidate_ids_list = []
        for ci in top_centroid_idxs:
            ids = self.load_inverted_list(int(ci))
            if ids.size > 0:
                candidate_ids_list.append(ids)
        del top_centroid_idxs
        
        if not candidate_ids_list:
            return []
        
        # Concatenate candidate IDs
        candidate_ids = np.concatenate(candidate_ids_list, axis=0)
        del candidate_ids_list
        n_candidates = candidate_ids.size
        
        # CRITICAL: Use VERY small batch size - process only ~2000 vectors at a time
        # This keeps memory usage under 512KB per batch (2000 * 64 * 4 bytes)
        batch_size = 2000
        
        # Use min-heap to track top_k results efficiently
        import heapq
        top_heap = []  # Format: (score, candidate_id)
        
        # Open database file for reading
        db_file = open(self.db_path, 'rb')
        
        try:
            for batch_start in range(0, n_candidates, batch_size):
                batch_end = min(batch_start + batch_size, n_candidates)
                batch_ids = candidate_ids[batch_start:batch_end]
                
                # Read vectors one-by-one to minimize memory
                for idx in batch_ids:
                    # Seek to position and read single vector
                    offset = int(idx) * DIMENSION * ELEMENT_SIZE
                    db_file.seek(offset)
                    vec_bytes = db_file.read(DIMENSION * ELEMENT_SIZE)
                    vec = np.frombuffer(vec_bytes, dtype=np.float32)
                    
                    # Normalize
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm > 0:
                        vec = vec / vec_norm
                    
                    # Compute score
                    score = float(np.dot(vec, qn))
                    
                    # Update heap
                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, (score, int(idx)))
                    elif score > top_heap[0][0]:
                        heapq.heapreplace(top_heap, (score, int(idx)))
                
                # Free batch_ids
                del batch_ids
                
        finally:
            db_file.close()
        
        del candidate_ids
        gc.collect()
        
        # Sort by score descending, then by ID for determinism
        top_heap.sort(key=lambda x: (-x[0], x[1]))
        result = [x[1] for x in top_heap]
        
        del top_heap
        return result
