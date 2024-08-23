"""

   A few plots to understand the Wishart 2D prior on Cov

"""

import numpy as np
import numpy.linalg as npl
import torch as th
import torch.linalg as thl

from pathlib import Path

from scipy.stats import chi2
import matplotlib.pyplot as plt


def viz_ellipse(prec):

    # -- extract geometry --
    # prec = th.from_numpy(prec)
    eigVals,eigVecs = th.linalg.eigh(prec)
    print("eig mult: ",eigVecs @ th.diag(eigVals) @ eigVecs.T)
    angle = th.atan2(eigVecs[0,[-1]],eigVecs[1,[-1]])
    print("-- eigVals,eigVecs --")
    print(eigVals)
    print(eigVecs)
    print("--> est angle: ",angle)

    confidence_level = 0.95/2.
    chi2_val = chi2.ppf(confidence_level, 2)
    radii = th.sqrt(chi2_val * eigVals)
    theta = th.linspace(0, 2 * th.pi, 100)
    ellipse = th.stack([radii[0]*th.cos(theta),radii[1]*th.sin(theta)],-1).double()
    rot = th.stack([th.stack([th.cos(angle),-th.sin(angle)],1),
                    th.stack([th.sin(angle),th.cos(angle)],1)],1)[0].double()
    print("-- rotation --")
    print(rot)

    # print(ellipse.shape,rot.shape)
    ellipse = th.einsum('ij,kj->ki', rot, ellipse)
    # print(ellipse.shape)

    # Plot the ellipses
    fig, ax = plt.subplots()
    ax.plot(ellipse[:,0],ellipse[:,1],
            label=f'90% Confidence Interval')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('90% Confidence Interval of 2D Gaussians')
    plt.grid(True)
    # ax.set_aspect("equal","box")
    ax = plt.gca()
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    # ax.set_aspect('equal', adjustable='box')
    plt.savefig("ellipse.png")
    plt.close("all")


def get_cov_mat(sigma_x,sigma_y,rho):
    sigma_xy = sigma_x * sigma_y
    cov = th.tensor([[sigma_x**2,rho*sigma_xy],
                    [rho*sigma_xy,sigma_y**2]])
    return cov

def main():

    # -- init --
    root = Path("./output/wishart_2d/")
    if not root.exists(): root.mkdir()

    # -- cov --
    sigma_x,sigma_y,rho = 2.,1.,0.
    cov_p = get_cov_mat(sigma_x,sigma_y,rho)
    prec_p = thl.pinv(cov_p)
    print(cov_p.shape)
    print(prec_p.shape)
    print(cov_p)
    print("prec_p: ")
    print(prec_p)


    theta = -th.tensor([th.pi/4.])
    print("--> known angle: ",theta)
    rot = th.tensor([[th.cos(theta),-th.sin(theta)],
                    [th.sin(theta),th.cos(theta)]])

    # print("--> rot: ")
    # print(rot)
    # ratio = 1.
    # L = th.diag(th.tensor([ratio, 1]))**2
    # print(rot @ L @ rot.T)
    # print(rot @ cov_p @ rot.T)
    # print(cov_p)
    # exit()
    # prec_c = thl.pinv(rot @ cov_p @ rot.T)
    prec_c = rot @ prec_p @ rot.T
    print("prec_c:")
    print(prec_c)

    viz_ellipse(prec_c)
    # viz_ellipse(prec_p)


if __name__ == "__main__":
    main()
