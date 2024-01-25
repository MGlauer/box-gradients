import torch
from torch import nn
import networkx as nx
import random
import tqdm
from matplotlib import pyplot as plt, colormaps
from matplotlib.patches import Rectangle
import math
import os

torch.autograd.set_detect_anomaly(True)

def normal(sigma, mu, x):
    v = (x-mu)/sigma
    return torch.exp(-0.5 * v*v)


class Points(nn.Module):
    def __init__(self, num_dim=2):
        super().__init__()
        self.embeddings = nn.Parameter(torch.rand(10000, num_dim, requires_grad=True))

    def forward(self, data):
        return self.embeddings[data]


class BoxWithMemberships(nn.Module):
    def __init__(self, num_labels, num_dim=2, eps=0.9, **kwargs):
        super().__init__(**kwargs)
        self.num_dim = num_dim

        self.num_labels = num_labels
        b = torch.rand((num_labels, num_dim, 2))
        b[:,:,1] = eps
        b += (1-eps)*torch.rand((num_labels, num_dim, 2))
        self.boxes = nn.Parameter(b, requires_grad=True)

    def forward(self, points, **kwargs):
        self.batch_size = points.shape[0]
        b = self.boxes.unsqueeze(0)

        l = torch.min(b, dim=-1)[0]
        r = torch.max(b, dim=-1)[0]

        #m = self._forward_gbmf(points, l, r)
        m = self._forward_membership(points, l, r)

        return self._soft_min_agg(m, dim=-1)

    def _forward_gbmf(self, points, left_corners, right_corners, **kwargs):
        widths = 0.1 * (right_corners - left_corners)
        return gbmf(points, left_corners+widths, right_corners-widths)


    def _soft_lukaziewisz_agg(self, memberships, dim=-1, scale=10):
        """
        This is a version of the Łukaziewish-T-norm using a modified softplus instead of max
        """
        return (
            1
            / scale
            * torch.log(
                1
                + torch.exp(
                    math.log(math.exp(scale) - 1)
                    * (torch.sum(memberships, dim=dim) - (memberships.shape[dim] - 1))
                )
            )
        )

    def _forward_membership(self, points, left_corners, right_corners, **kwargs):
        widths = 0.1 * (right_corners - left_corners)
        max_distance_per_dim = nn.functional.relu(left_corners - points + widths) + nn.functional.relu(points - right_corners + widths)

        m = normal(widths, 0, max_distance_per_dim)
        return m

    def prod_agg(self, memberships, dim=-1):
        return torch.relu(torch.sum(memberships, dim=dim)-(memberships.shape[dim]-1))

    def min_agg(self, memberships, dim=-1):
        return torch.relu(torch.sum(memberships, dim=dim)-(memberships.shape[dim]-1))

    def _soft_min_agg(self, memberships, dim=-1):
        return torch.sum(torch.softmax(-memberships, dim=dim)*memberships, dim=dim)

    def lukaziewisz_agg(self, memberships, dim=-1):
        return torch.relu(torch.sum(memberships, dim=dim)-(memberships.shape[dim]-1))

    def crisp_forward(self, points):
        self.batch_size = points.shape[0]
        b = self.boxes.unsqueeze(0)
        l = torch.min(b, dim=-1)[0]
        r = torch.max(b, dim=-1)[0]
        return torch.min((l <= points) * (points <= r), dim=-1)[0]

    def lukaziewisz_agg(self, memberships, dim=-1):
        return torch.relu(torch.sum(memberships, dim=dim)-(memberships.shape[dim]-1))



def calculate_gradients(boxes, raw_points, idx, target):

    b = boxes.unsqueeze(0)
    points = raw_points[idx]
    target = target.unsqueeze(-1)
    l, lind = torch.min(b, dim=-1)
    r, rind = torch.max(b, dim=-1)




    width = (r-l)/2
    r_fp = r + 0.1*width
    r_fn = r - 0.1 * width

    l_fp = l - 0.1*width
    l_fn = l + 0.1 * width

    inside = ((l<points)*(points<r))
    inside_fp = (l_fp < points) * (points < r_fp)
    inside_fn = (l_fn < points) * (points < r_fn)

    fn_per_dim = ~inside_fn * target
    fp_per_dim = inside_fp * (1-target)

    false_per_dim = fn_per_dim + fp_per_dim
    number_of_false_dims = torch.sum(false_per_dim, dim=-1, keepdim=True)

    dl = torch.abs(l-points)
    dr = torch.abs(r-points)

    closer_to_l_than_r = dl < dr

    r_scale_fp = number_of_false_dims*torch.rand_like(fp_per_dim)*(fp_per_dim * ~closer_to_l_than_r)
    l_scale_fp = number_of_false_dims*torch.rand_like(fp_per_dim)*(fp_per_dim * closer_to_l_than_r)

    r_scale_fn = number_of_false_dims*torch.rand_like(fn_per_dim)*(fn_per_dim * ~closer_to_l_than_r)
    l_scale_fn = number_of_false_dims*torch.rand_like(fn_per_dim)*(fn_per_dim * closer_to_l_than_r)

    r_loss = torch.mean(torch.sum(torch.abs(r_scale_fp * (r_fp - points)), dim=(1,2)) + torch.sum(torch.abs(r_scale_fn * (r_fn - points)), dim=(1,2)))
    l_loss = torch.mean(torch.sum(torch.abs(l_scale_fp * (l_fp - points)), dim=(1,2)) + torch.sum(torch.abs(l_scale_fn * (l_fn - points)), dim=(1,2)))
    return l_loss + r_loss

def plot_boxes(ax, boxes, box_colors):
    b = boxes[:, :2, :]
    l = torch.min(b, dim=-1)[0]
    r = torch.max(b, dim=-1)[0]
    widths = r-l
    for i in range(boxes.shape[0]):
        ax.add_patch(Rectangle(l[i].detach().numpy(), widths[i, 0].item(), widths[i, 1].item(), linewidth=1, edgecolor=box_colors.colors[i], facecolor='none'))
    xl,yl = zip(*l.detach().numpy())
    ax.plot(xl, yl, ".")
    xr, yr = zip(*r.detach().numpy())
    ax.plot(xr, yr, ".")

def gbmf(x, l, r, b=6):
    a = (r-l)+1e-3
    c = l+(r-l)/2
    return 1 / (1 + (torch.abs((x - c) / a) ** (2 * b)))

def main():
    os.makedirs("plt", exist_ok=True)
    num_classes = 10

    box_colors = colormaps["tab10"]

    individuals = 10000
    G = nx.gnp_random_graph(10, 0.5, directed=True, seed=5)
    ontology = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
    nx.draw(ontology, pos=nx.drawing.spiral_layout(ontology), node_color=box_colors.colors)
    plt.savefig("plt/ontology.png")
    plt.close()
    #ontology = nx.random_tree(num_classes, create_using=nx.DiGraph)
    trans_ontology = nx.transitive_closure(ontology)
    classes = list(ontology.nodes)
    dataset = []
    for n in range(individuals):
        n_parents = random.randint(1,4)
        parents = random.sample(ontology.nodes, n_parents)
        preds = {p2 for p in parents for p2 in trans_ontology.predecessors(p)}
        preds = preds.union(parents)
        y = torch.tensor([i in preds for i in classes]).float()
        dataset.append((torch.tensor([n]),y))

    points = Points()
    model = BoxWithMemberships(num_classes)

    points_optimizer = torch.optim.Adam(points.parameters(), lr=1e-3)
    box_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    Xtrain, ytrain = zip(*dataset)

    Xtrain = torch.stack(Xtrain)
    ytrain = torch.stack(ytrain)

    n_epochs = 300  # number of epochs to run
    batch_size = 100  # size of each batch
    batches_per_epoch = len(Xtrain) // batch_size
    previous_loss= "-"
    for epoch in range(n_epochs):
        running_loss = 0
        perm = torch.randperm(Xtrain.size()[0])
        Xtrain = Xtrain[perm]
        ytrain = ytrain[perm]
        for i in tqdm.tqdm(range(batches_per_epoch), desc=f"epoch: {epoch}, loss: {previous_loss}"):
            points_optimizer.zero_grad()
            box_optimizer.zero_grad()

            start = i * batch_size
            # take a batch
            Xbatch = Xtrain[start:start + batch_size]
            ybatch = ytrain[start:start + batch_size]
            loss = calculate_gradients(model.boxes, points.embeddings, Xbatch, ybatch)
            loss.backward()
            running_loss += loss
            points_optimizer.step()
            box_optimizer.step()


        fig, ax = plt.subplots()
        model.eval()

        pts = points(Xtrain)
        y_pred = model.crisp_forward(pts)
        emb = points(Xtrain[:,0]).detach().numpy()[:,:2]
        correctness = torch.mean((ytrain - y_pred.float())**2, dim=-1)**0.5
        colors = colormaps["cool"](correctness)
        model.train()
        ax.scatter(*zip(*emb), s=0.1, marker=",", color=colors)
        #ax.plot(*zip(*emb), "b,")

        plot_boxes(ax, model.boxes, box_colors)
        plt.savefig(f"plt/epoch{epoch:03d}.png")
        plt.close()
        previous_loss = running_loss/batches_per_epoch


main()