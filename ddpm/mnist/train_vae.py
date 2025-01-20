import os, sys

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader

from einops import rearrange

from torchvision.utils import save_image

from purias_utils.util.logging import configure_logging_paths

from ddpm.mnist.vae import MNISTVAE, VAELatentClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


CLASSIFIER_WEIGHT = 0.0
COVAR_SCALER = 0.6

z_dim = int(sys.argv[1])
num_classes = 10

mnist_path = "/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/data/mnist"
logging_directory = f"/homes/pr450/repos/research_projects/error_modelling_torus/results_link_sampler/mnist/vae/z{z_dim}_run"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=mnist_path, train=True, transform=transform)
test_dataset = datasets.MNIST(root=mnist_path, train=False, transform=transform)
dataset = ConcatDataset([train_dataset, test_dataset])

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

vae = MNISTVAE(z_dim=z_dim).cuda()
classifier = VAELatentClassifier(z_dim=z_dim).cuda()

optimizer = optim.Adam(list(vae.parameters()) + list(classifier.parameters()))

colours = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
    "#00fff7",  # cyan
]


# return reconstruction error + KL divergence losses
def get_vae_elbo(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def get_classification_loss(pred_logits, classes):
    return F.cross_entropy(pred_logits, classes.to(pred_logits.device), reduction="sum")


def train(epoch, dataloader, print_dest, classification_loss_weight=CLASSIFIER_WEIGHT):
    vae.train()
    train_loss = 0
    class_loss = 0

    running_class_means = torch.zeros(num_classes, z_dim).cuda()
    running_class_outer = torch.zeros(num_classes, z_dim, z_dim).cuda()
    class_counts = torch.zeros(num_classes).cuda()

    for batch_idx, (data, class_labels) in enumerate(dataloader):
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        classifcation_logits = classifier(mu)
        loss = get_vae_elbo(recon_batch, data, mu, log_var)
        classification_loss = get_classification_loss(
            classifcation_logits, class_labels
        )

        for one_mu, class_idx in zip(mu, class_labels):
            running_class_means[class_idx] = running_class_means[class_idx] + one_mu
            running_class_outer[class_idx] = running_class_outer[
                class_idx
            ] + torch.einsum("i,j->ij", one_mu, one_mu)
            class_counts[class_idx] += 1

        (loss + classification_loss_weight * classification_loss).backward()
        train_loss += loss.item()
        class_loss += classification_loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            with open(print_dest, "a") as f:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClassifier Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(dataloader.dataset),
                        100.0 * batch_idx / len(dataloader),
                        loss.item() / len(data),
                        classification_loss.item() / len(data),
                    ),
                    file=f,
                )

    running_class_means = running_class_means.detach() / class_counts[:, None]
    running_class_outer = running_class_outer.detach() / class_counts[:, None, None]

    with open(print_dest, "a") as f:
        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(dataloader.dataset)
            ),
            file=f,
        )

    return (
        mu.detach(),
        class_labels,
        train_loss,
        class_loss / len(dataloader.dataset),
        running_class_means,
        running_class_outer,
    )


[training_print_path], logging_directory, _ = configure_logging_paths(
    logging_directory, log_suffixes=["train"], index_new=True
)


all_losses = []
all_class_losses = []

for epoch in range(1, 51):

    (
        embedding_means,
        class_labels,
        latest_loss,
        latest_class_loss,
        empirical_class_means,
        empirical_class_outer,
    ) = train(epoch, dataloader, training_print_path)
    torch.save(vae.state_dict(), os.path.join(logging_directory, "vae_state.mdl"))
    torch.save(
        classifier.state_dict(), os.path.join(logging_directory, "classifier_state.mdl")
    )
    all_losses.append(latest_loss)
    all_class_losses.append(latest_class_loss)

    with torch.no_grad():
        # Visualise latent space
        tsne_num_dim = 3
        fig, axes = plt.subplots(tsne_num_dim, tsne_num_dim, figsize=(20, 20))
        # pca_embedding_means = PCA(n_components=z_dim, whiten=True).fit_transform(embedding_means.cpu().numpy())
        pca_embedding_means = TSNE(n_components=tsne_num_dim).fit_transform(
            embedding_means.cpu().numpy()
        )
        for i, axrow in enumerate(axes):
            for j, ax in enumerate(axrow):
                if i <= j:
                    continue
                ax.scatter(
                    pca_embedding_means[:, i],
                    pca_embedding_means[:, j],
                    c=class_labels,
                    cmap=ListedColormap(colours),
                )
        fig.savefig(os.path.join(logging_directory, "class_pca.png"))

        # Visualise new samples
        sqrt_batch = 8
        display_batch = int(sqrt_batch**2)
        z = torch.randn(display_batch, z_dim).cuda()
        sample = (
            vae.decoder(z).cuda().view(display_batch, 28, 28)
        )  # [B, D^2] -> [B, D, D]
        tiled_sample = rearrange(
            sample, "(b1 b2) i j -> (b1 i) (b2 j)", b1=sqrt_batch, b2=sqrt_batch
        )
        save_image(
            tiled_sample, os.path.join(logging_directory, "random_sample") + ".png"
        )

        # Visualise samples generated from empirical means
        mean_samples = vae.decoder(empirical_class_means).view(
            num_classes, 28, 28
        )  # [C, D^2] -> [C, D, D]
        tiled_mean_samples = rearrange(mean_samples, "c i j -> i (c j)", c=num_classes)
        save_image(
            tiled_mean_samples,
            os.path.join(logging_directory, "mean_latent_sample") + ".png",
        )

        # Visualise samples generated from empirical distributions
        mean_squared = torch.einsum(
            "bi,bj->bij", empirical_class_means, empirical_class_means
        )
        class_covars = empirical_class_outer - mean_squared
        class_distribution = torch.distributions.MultivariateNormal(
            loc=empirical_class_means, covariance_matrix=COVAR_SCALER * class_covars
        )
        class_z = class_distribution.sample((sqrt_batch,))
        class_sample = vae.decoder(class_z).cuda().view(sqrt_batch, num_classes, 28, 28)
        tiled_class_sample = rearrange(
            class_sample, "b c i j -> (b i) (c j)", c=num_classes
        )
        save_image(
            tiled_class_sample,
            os.path.join(logging_directory, "class_latent_sample") + ".png",
        )

        # Visualise loss
        fig, axes = plt.subplots(2, figsize=(5, 10))
        axes[0].plot(all_losses)
        axes[0].set_title("VAE ELBO")
        axes[1].plot(all_class_losses)
        axes[1].set_title("Classification loss (averaged for interpretability)")
        fig.savefig(os.path.join(logging_directory, "losses.png"))

        # Save moments for each class
        torch.save(
            {
                "empirical_class_means": empirical_class_means,
                "class_covars": class_covars,
                "covar_scaler": COVAR_SCALER,
            },
            os.path.join(logging_directory, "vae_latent_class_moments.data"),
        )

        plt.cla()
