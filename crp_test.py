import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

###### SET UP DATASET

# Define parameters for 4 Gaussian components
means = np.array([
    [2, 2],    # Cluster 1
    [-2, -2],  # Cluster 2
    [2, -2],   # Cluster 3
    [-2, 2]    # Cluster 4
])

weights = np.array([0.1, 0.15, 0.25, 0.5])

covariances = np.array([
    [[0.5, 0.2], [0.2, 0.5]],  # Cluster 1
    [[0.7, -0.3], [-0.3, 0.7]], # Cluster 2
    [[0.6, 0.1], [0.1, 0.6]],  # Cluster 3
    [[0.8, -0.2], [-0.2, 0.8]]  # Cluster 4
])

num_data = 400
num_comps, num_dim = means.shape

# Generate dataset
components = np.random.choice(range(num_comps), num_data, p=weights)
data = np.zeros([num_data, num_dim])
for i in range(4):
    comp_mask = (components == i)
    data[comp_mask] = np.random.multivariate_normal(means[i], covariances[i], comp_mask.sum())

data = torch.tensor(np.vstack(data), dtype = torch.float32)
labels = torch.tensor(np.array(components), dtype = torch.float32)




###### RANDOM INIT OF CRP
def reorder_clusters(Z: torch.Tensor, *arrs: torch.Tensor):
    """
    Requires:
    - Z of shape [num data] (indices)
    - all of arrs start with [num clusters]

    Returns:
    - Z_new of shape [num data] (indices), renamed for popularity
    - num_used_clusters - number of clusters that had at least one assigned data point
    - arrs_new - arrs reordered by popularity, moving unassigned clusters to the end
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

    return Z_new, num_used_clusters, arrs_new



def get_multivariate_student_likelihoods(
    X, df, mu, Sigma
):
    """
    Requires:
    X of shape [B, D]
    df of shape [C]
    mu of shape [C, D]
    Sigma of shape [C, D, D]

    Returns:
    likelihoods of shape [B, C]
    """
    D = mu.shape[1]

    X = X.unsqueeze(1)          # [B, 1, D]
    df = df.unsqueeze(0)        # [1, C]
    mu = mu.unsqueeze(0)        # [1, C, D]

    exponent = 0.5 * (df + D)
    coeff_term = (
        torch.lgamma(exponent) 
        - torch.lgamma(df / 2.0) 
        - 0.5 * (D * torch.log(df * torch.pi) - torch.logdet(Sigma))
    )    # [1, C]

    residual = X - mu   # [B, C, D]
    inv_Sigma = torch.linalg.inv(Sigma)
    exponent_term = 1.0 + torch.einsum('bci,cij,bcj->bc', residual, inv_Sigma, residual) / df     # [B, C]

    total_llh = coeff_term - (exponent * exponent_term)

    return total_llh.exp()




# Cluster parameters
init_crp_comps = 16
max_crp_comps = 50
alpha_parameter = torch.tensor([1.0])
num_iterations = 100

# NIW prior parameters
# niw_mu_0 = torch.zeros(num_dim)   # Kept as zero throughout! makes things easier...
niw_kappa = 1.0
niw_Psi = torch.eye(num_dim)
niw_nu = torch.tensor([float(num_dim)])
prior_student_sigma = ((niw_kappa + 1.0) / niw_kappa * (niw_nu - num_dim + 1)) * niw_Psi

# Initialise clusters and reorder so that the richest are at the start
num_clusters = init_crp_comps
cluster_assignments = torch.randint(0, init_crp_comps, [num_data])


# Start keeping track of important stats for each cluster, with space to track them for all clusters we allow
# nu and kappa easy to get based on cluster_counts
cluster_counts = torch.zeros(max_crp_comps)  # N
cluster_sums = torch.zeros(max_crp_comps, num_dim)  # N * x^bar --> more convinient than just midpoint x^bar
cluster_outer_of_sums = torch.zeros(max_crp_comps, num_dim, num_dim)   # N * N * x^bar x^bar^T
cluster_sum_of_outers = torch.zeros(max_crp_comps, num_dim, num_dim)   
cluster_scatters = torch.zeros(max_crp_comps, num_dim, num_dim)

cluster_idx, cluster_assignment_counts = cluster_assignments.unique(return_counts = True)
cluster_counts[:len(cluster_assignment_counts)] = cluster_assignment_counts

for c_idx in cluster_idx:
    relevant_data = data[cluster_assignments == c_idx]
    new_cluster_sum = relevant_data.sum(0)
    cluster_residuals = (relevant_data - relevant_data.mean(0).unsqueeze(0))
    cluster_sums[c_idx] = new_cluster_sum
    cluster_outer_of_sums[c_idx] = torch.einsum('i,j->ij', new_cluster_sum, new_cluster_sum)
    cluster_sum_of_outers[c_idx] = torch.einsum('bi,bj->ij', relevant_data, relevant_data)
    cluster_scatters[c_idx] = (
        cluster_sum_of_outers[c_idx] - 
        (cluster_outer_of_sums[c_idx] / cluster_counts[c_idx])
    )

cluster_assignments, num_clusters, (cluster_counts, cluster_sums, cluster_outer_of_sums, cluster_sum_of_outers, cluster_scatters) = \
        reorder_clusters(cluster_assignments, cluster_counts, cluster_sums, cluster_outer_of_sums, cluster_sum_of_outers, cluster_scatters)

# Iterate until some form of convergence...
for _ in tqdm(range(num_iterations)):

    num_reassignments = 0
        
    for n_idx in range(num_data):

        data_vector = data[[n_idx]]  # [1, D]
        current_cluster_assignment = cluster_assignments[n_idx] # scalar
        
        # Generate a temporary cache which is updated every time we exclude one datapoint, and replaces
        # the main stats when a datapoint actually changes clusters
        temp_cluster_counts = cluster_counts.clone()                    # [C]
        temp_cluster_sums = cluster_sums.clone()                        # [C, D]
        temp_cluster_outer_of_sums = cluster_outer_of_sums.clone()      # [C, D, D]
        temp_cluster_sum_of_outers = cluster_sum_of_outers.clone()      # [C, D, D]
        temp_cluster_scatters = cluster_scatters.clone()      # [C, D, D]

        # Update temporary cache to exclude this datapoint
        temp_cluster_counts[current_cluster_assignment] = cluster_counts[current_cluster_assignment] - 1    
        temp_cluster_sums[current_cluster_assignment] = cluster_sums[current_cluster_assignment] - data_vector
        temp_cluster_outer_of_sums[current_cluster_assignment] = torch.einsum('i,j->ij', temp_cluster_sums[current_cluster_assignment], temp_cluster_sums[current_cluster_assignment])
        temp_cluster_sum_of_outers[current_cluster_assignment] = cluster_sum_of_outers[current_cluster_assignment] - torch.einsum('i,j->ij', data_vector[0], data_vector[0])
        temp_cluster_scatters[current_cluster_assignment] = (
            temp_cluster_sum_of_outers[current_cluster_assignment] - 
            (temp_cluster_outer_of_sums[current_cluster_assignment] / temp_cluster_counts[current_cluster_assignment])
        )

        # Calculate NIW posterior parameters for all clusters
        cluster_post_kappas = temp_cluster_counts + niw_kappa  # [C]
        cluster_post_nu = temp_cluster_counts + niw_nu          # [C]
        cluster_post_mu = temp_cluster_sums / cluster_post_nu.unsqueeze(-1)  # [C, D]
        cluster_post_psi_last_term = temp_cluster_outer_of_sums / (temp_cluster_sums * cluster_post_kappas[:,None])[:,None] # [C, D, D] - x^bar x^bar^T * N / kappa = N * N * x^bar x^bar^T / (N * kappa)
        cluster_post_psi = niw_Psi.unsqueeze(0) + temp_cluster_scatters + cluster_post_psi_last_term    # [C, D, D]

        # Calculate the predictive distribution parameters for each cluster (using t distribution), and the prior weight provided by the DP (unweighted)
        # Extend this to allow a new cluster if allowed
        # XXX: track these two so that inverse and determinant can be cached also!
        cluster_student_degs = cluster_post_nu - num_dim + 1                    # [C]
        cluster_student_sigma = ((cluster_post_kappas + 1) / (cluster_post_kappas * cluster_student_degs))[:,None,None] * cluster_post_psi  # [C, D, D]
        
        if num_clusters < max_crp_comps:
            # cluster_student_degs will already have the prior NIW degrees of freedom at index num_clusters
            cluster_student_sigma[num_clusters] = prior_student_sigma  # still [C], but possibly with a new cluster added (not nan)
            num_clusters_to_consider = num_clusters + 1     # C'
            stick_breaking_post_weights = temp_cluster_counts[:num_clusters_to_consider]
            assert stick_breaking_post_weights[-1] == 0.0
            stick_breaking_post_weights[-1] = alpha_parameter
        else:
            num_clusters_to_consider = num_clusters         # C'
            stick_breaking_post_weights = temp_cluster_counts[:num_clusters]    # [C']
        
        # Calculate probability of assigning to each component
        multivariate_student_likelihoods = get_multivariate_student_likelihoods(
            data_vector, cluster_student_degs[:num_clusters_to_consider],
            cluster_post_mu[:num_clusters_to_consider], cluster_student_sigma[:num_clusters_to_consider]
        )   # [1, C']
        dp_weighted_multivariate_student_likelihoods = stick_breaking_post_weights[None] * multivariate_student_likelihoods
        dp_weighted_multivariate_student_likelihoods = dp_weighted_multivariate_student_likelihoods / dp_weighted_multivariate_student_likelihoods.sum(-1, keepdim = True)

        # Select new place for datapoint
        assignment_cdf = dp_weighted_multivariate_student_likelihoods.cumsum(-1)    # [1, C']
        rand_u = torch.rand(len(assignment_cdf), 1) # [1, C']
        proposed_cluster_assignment = (rand_u > assignment_cdf).sum(-1) # [1]
        proposed_cluster_assignment = proposed_cluster_assignment.item()    # Scalar!

        # Tricky logic...
        # If no change, then nothing happens - we can continue to use the stats we have been accumulating
        if proposed_cluster_assignment == current_cluster_assignment:
            pass
        
        # If assigned to a new cluster, we have to initiaise stats for that cluster...
        elif proposed_cluster_assignment == num_clusters:

            # Make the removal from the previous cluster permanent
            cluster_counts[current_cluster_assignment] = temp_cluster_counts[current_cluster_assignment]
            cluster_sums[current_cluster_assignment] = temp_cluster_sums[current_cluster_assignment]
            cluster_outer_of_sums[current_cluster_assignment] = temp_cluster_outer_of_sums[current_cluster_assignment]
            cluster_sum_of_outers[current_cluster_assignment] = temp_cluster_sum_of_outers[current_cluster_assignment]
            cluster_scatters[current_cluster_assignment] = temp_cluster_scatters[current_cluster_assignment]

            cluster_counts[num_clusters] = 1
            cluster_sums[num_clusters] = data_vector
            cluster_outer_of_sums[num_clusters] = torch.einsum('bi,bj->ij', data_vector, data_vector)
            cluster_sum_of_outers[num_clusters] = cluster_outer_of_sums[num_clusters]   # Same for one datapoint
            cluster_scatters[num_clusters] = (
                cluster_sum_of_outers[num_clusters] - (cluster_outer_of_sums[num_clusters] / cluster_counts[num_clusters])
            )

            num_clusters += 1

            num_reassignments += 1

        # If assigned to another existing cluster, we have to update the stats of that cluster too
        else:

            # Make the removal from the previous cluster permanent
            cluster_counts[current_cluster_assignment] = temp_cluster_counts[current_cluster_assignment]
            cluster_sums[current_cluster_assignment] = temp_cluster_sums[current_cluster_assignment]
            cluster_outer_of_sums[current_cluster_assignment] = temp_cluster_outer_of_sums[current_cluster_assignment]
            cluster_sum_of_outers[current_cluster_assignment] = temp_cluster_sum_of_outers[current_cluster_assignment]
            cluster_scatters[current_cluster_assignment] = temp_cluster_scatters[current_cluster_assignment]

            cluster_counts[proposed_cluster_assignment] = cluster_counts[proposed_cluster_assignment] + 1    # [C]
            cluster_sums[proposed_cluster_assignment] = cluster_sums[proposed_cluster_assignment] + data_vector   # [C, D]
            cluster_outer_of_sums[proposed_cluster_assignment] = torch.einsum('i,j->ij', cluster_sums[proposed_cluster_assignment], cluster_sums[proposed_cluster_assignment])
            cluster_sum_of_outers[proposed_cluster_assignment] = cluster_sum_of_outers[proposed_cluster_assignment] + torch.einsum('i,j->ij', data_vector[0], data_vector[0])
            cluster_scatters[num_clusters] = (
                cluster_sum_of_outers[proposed_cluster_assignment] - (cluster_outer_of_sums[proposed_cluster_assignment] / cluster_counts[proposed_cluster_assignment])
            )
            num_reassignments += 1

        cluster_assignments[n_idx] = proposed_cluster_assignment

        cluster_assignments, num_clusters, (cluster_counts, cluster_sums, cluster_outer_of_sums, cluster_sum_of_outers, cluster_scatters) = \
            reorder_clusters(cluster_assignments, cluster_counts, cluster_sums, cluster_outer_of_sums, cluster_sum_of_outers, cluster_scatters)
        if num_clusters < max_crp_comps:
            assert cluster_counts[num_clusters] == 0


    print(num_reassignments, num_clusters)
    if num_reassignments == 0:
        import pdb; pdb.set_trace()


    # Plot the generated data
    plt.close('all')
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)

    n_std = 2.0

    for i_c in range(num_clusters):
        mean = cluster_post_mu[i_c]
        cov = cluster_student_sigma[]

        plt.scatter(*mean[None].T, color = 'black')

        eigenvalues, eigenvectors = np.linalg.eigh()  # Eigen decomposition
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))  # Get ellipse angle
        width, height = 2 * n_std * np.sqrt(eigenvalues)  # Scale by standard deviation
        
        # Create an ellipse
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=edgecolor, facecolor='none', linestyle='--')
        plt.add_patch(ellipse)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Generated Data from a Mixture of 4 Gaussians")
    plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i+1}" for i in range(4)])
    plt.grid(True)
    plt.savefig('CRP.png')
