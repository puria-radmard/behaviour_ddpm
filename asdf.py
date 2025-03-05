import torch

def reorder_clusters(Z: torch.Tensor, *arrs: torch.Tensor):
    """
    Requires:
    - Z of shape [num data] (indices)
    - all of arrs start with [num clusters]

    Returns:
    - Z_new of shape [num data] (indices), renamed for popularity
    - arrs_new - arrs reordered by popularity, moving unassigned clusters to the end
    - num_used_clusters - number of clusters that had at least one assigned data point
    """

    num_clusters = arrs[0].shape[0]  # Total clusters in arrs
    unique_labels, counts = torch.unique(Z, return_counts=True)  # Find active clusters
    num_used_clusters = len(unique_labels)  # Count of clusters actually used

    # Step 1: Sort by frequency (most frequent first)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_labels = unique_labels[sorted_indices]  # These clusters exist in Z

    # Step 2: Find unused clusters (not in Z)
    all_clusters = torch.arange(num_clusters)  # Full cluster index range
    unused_clusters = all_clusters[~torch.isin(all_clusters, unique_labels)]  # Not assigned in Z

    # Step 3: Construct final ordering (used clusters first, then unused)
    final_order = torch.cat((sorted_labels, unused_clusters))  # Popular first, unused last

    # Step 4: Create mapping {old_label -> new_label}
    new_label_map = {old.item(): new_idx for new_idx, old in enumerate(final_order)}

    # Step 5: Apply mapping to Z
    Z_new = torch.tensor([new_label_map[label.item()] for label in Z])

    # Step 6: Reorder arrs
    arrs_new = tuple(arr[final_order] for arr in arrs)

    return Z_new, arrs_new, num_used_clusters

# Example Usage
Z = torch.tensor([4, 5, 1, 0, 3, 2, 4, 1, 1, 0, 3, 4, 4])  # Cluster assignments

cluster_means = torch.tensor([  # num_clusters x dim
    [0.1, 0.2],  # Cluster 0
    [1.1, 1.2],  # Cluster 1
    [2.1, 2.2],  # Cluster 2
    [3.1, 3.2],  # Cluster 3
    [4.1, 4.2],  # Cluster 4
    [5.1, 5.2],  # Cluster 5
    [6.1, 6.2],  # Extra cluster 6 (unused)
    [7.1, 7.2]   # Extra cluster 7 (unused)
])

# Apply function
Z_new, (cluster_means_new,), num_used_clusters = reorder_clusters(Z, cluster_means)

print("Original Z:", Z.tolist())
print("Reindexed Z:", Z_new.tolist())
print("Original Cluster Means:\n", cluster_means)
print("Reordered Cluster Means:\n", cluster_means_new)
print("Number of Used Clusters:", num_used_clusters)
