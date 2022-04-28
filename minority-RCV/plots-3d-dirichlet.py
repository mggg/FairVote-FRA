
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
from math import gamma
from operator import mul
from functools import reduce

def plot_mesh(corners):
    """Subdivide the triangle into a triangular mesh and plot the original and subdivided triangles."""
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)
    
    plt.figure(figsize=(6, 4))
    for i, mesh in enumerate((triangle, trimesh)):
        plt.subplot(1, 2, i+1)
        plt.triplot(mesh)
        plt.axis('off')
        plt.axis('equal')    
        
        
class Dirichlet:
    """Define the Dirichlet distribution with vector parameter alpha."""
    def __init__(self, alpha):
        
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])
    
    def pdf(self, x):
        """Returns pdf value for `x`. """
        return self._coef * reduce(mul, [xx ** (aa-1) for (xx, aa) in zip(x, self._alpha)])
        
        
class PlotDirichlet:
    """
    Plot the Dirichlet distribution as a contour plot on a 2-Simplex.
    """
    def __init__(self, corners):
        self._corners = corners
        self._triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        # Midpoints of triangle sides opposite of each corner
        self._midpoints = [(corners[(i+1) % 3] + corners[(i+2) % 3]) / 2.0 for i in range(3)]
        
    def xy2bc(self, xy, tol=1.e-3):
        """Map the x-y coordinates of the mesh vertices to the simplex coordinate space (aka barycentric coordinates).
        Here we use a simple method that uses vector algebra. For some values of alpha, calculation of the Dirichlet pdf 
        can become numerically unstable at the boundaries of the simplex so our conversion function will take an optional 
        tolerance that will avoid barycentric coordinate values directly on the simplex boundary.        
        """
        s = [(self._corners[i] - self._midpoints[i]).dot(xy - self._midpoints[i]) / 0.75 for i in range(3)]
        return np.clip(s, tol, 1.0-tol)
        
    def draw_pdf_contours(self, ax, dist, label=None, nlevels=200, subdiv=8, **kwargs):
        """Draw pdf contours for a Dirichlet distribution"""
        # Subdivide the triangle into a triangular mesh
        refiner = tri.UniformTriRefiner(self._triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)
        
        # convert to barycentric coordinates and compute probabilities of the given distribution 
        pvals = [dist.pdf(self.xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    
        ax.tricontourf(trimesh, pvals, nlevels, cmap="plasma", **kwargs)
        # plt.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75**0.5)

        # Get the left, right, and top corners.
        adj=1/22
        lc = (0,0-adj)
        rc = (1,0-adj)
        tc = (1/2, (math.sqrt(3)+adj)/2)

        # Add text at the corners.
        ax.text(*lc, fr"${label}_1$", ha="right")
        ax.text(*tc, fr"${label}_2$", ha="center")
        ax.text(*rc, fr"${label}_3$", ha="left")

        ax.axis('off') 
        return ax

        
if __name__ == '__main__':
    # Use LaTeX. Create figures and axes.
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r'\usepackage{nicefrac}',
        "font.family": "serif",
        "font.serif": "New Century Schoolbook"
    })

    # Set color map.
    kwargs = dict(cmap="plasma")

    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    plot_dirichlet = PlotDirichlet(corners)
   
    fig, axes = plt.subplots(1, 3, figsize=(7, 2))

    # Set the offset for titles.
    offset = 1/6

    # Set the comically close-to-uniform inducing concentration parameter.
    c = np.float64(np.float64(1) - np.float64(10**-2))

    ax = axes[0]
    alpha = (c, c, c)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, label="C")
    ax.text(1/2, -offset, r"$\alpha < 1$", ha="center")
    ax.set_aspect("equal")

    ax = axes[1]
    alpha = (1, 1, 1)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, label="C")
    ax.text(1/2, -offset, r"$\alpha=1$", ha="center")
    ax.set_aspect("equal")

    ax = axes[2]
    alpha = (2, 2, 2)
    dist = Dirichlet(alpha)
    ax = plot_dirichlet.draw_pdf_contours(ax, dist, label="C")
    ax.text(1/2, -offset, r"$\alpha > 1$", ha="center")
    ax.set_aspect("equal")

    # plt.show()
    fig.tight_layout(pad=2)
    plt.savefig('./output/figures/dirichlet.png', bbox_inches='tight', dpi=600)
