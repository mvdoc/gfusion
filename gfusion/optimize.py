"""Module for optimization routines"""
import numpy as np

from .due import due, Doi, BibTeX


@due.dcite(BibTeX("""@article{chen2011projection,
           title={Projection onto a simplex},
           author={Chen, Yunmei and Ye, Xiaojing},
           journal={arXiv preprint arXiv:1101.6081},
           year={2011}
           }"""),
           description="Efficient algorithm for projection onto a simplex")
def simplex_projection(y):
    """Projection of a n-dimensional vector onto the simplex Dn
    Dn = { x : x n-dim, 0 <= x <= 1, sum(x) = 1}

    Algorithm by Xiaojing Ye, based on matlab implementation

    see http://arxiv.org/abs/1101.6081

    Parameters
    ---------
    y : np.ndarray (n_features, )

    Returns
    -------
    x : np.ndarray (n_features, )
        projection onto the simplex Dn
    """
    if not isinstance(y, (np.ndarray, list)):
        y = np.array([y])

    n_features = len(y)
    if n_features == 0:
        raise ValueError('y cannot be empty')

    sorted_y = np.sort(y)[::-1]
    tmpsum = 0.
    bget = False

    for i in range(n_features - 1):
        tmpsum += sorted_y[i]
        tmax = (tmpsum - 1.) / (i + 1)

        if tmax >= sorted_y[i + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + sorted_y[-1] - 1) / n_features

    x = np.max(np.vstack((np.asarray(y) - tmax,
                          np.zeros(n_features))), axis=0)
    return x





