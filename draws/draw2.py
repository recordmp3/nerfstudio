import imageio
import numpy as np
import matplotlib.pyplot as plt


def draw(x, y, x_label, y_label, title=None, save_path=None, label=None):

    plt.style.use("ggplot")
    if label is None:
        label = [None for i in range(len(y))]
    for i in range(len(y)):
        plt.plot(x, y[i], linewidth=1, label=label[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    if label is not None:
        plt.legend(loc="best", numpoints=1, fancybox=True)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path + "/" + title + ".png", dpi=120)  # bbox_inches='tight')
    plt.close()


def compute_loss(x, y):

    return ((x - y) ** 2).mean()


f = open("/home/zt15/projects/nerfstudio/theta.txt", "r")
thetas = f.read().split(" ")
print(len(f.read().split(" ")))

f = open("/home/zt15/projects/nerfstudio/rgb.txt", "r")
rgbs = f.read().split(" ")

x = []
theta = []
loss = []
xx = []
print(len(rgbs), len(thetas))
for i in range(0, len(rgbs) - 1):
    x.append(i)
    id = int(len(thetas) / len(rgbs) * i)
    theta.append(float(thetas[id]))
    loss.append(float(rgbs[i]))


draw(x, [theta], "iters", "theta")
draw(x, [loss], "iters", "loss")
# draw(xx, [loss], "image id", "loss")
