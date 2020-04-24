import numpy as np

def NMF(V, components, iter, e):
    """
    Nonnegative Matrix Factorization
    :param V: original matrix
    :param components: how many features
    :param iter: iteration numbers
    :param e: error threshold
    :return:
    """

    V = V.T
    # get dimension
    m, n = V.shape
    # Randomize two matrix
    W = np.random.random((m, components))
    H = np.random.random((components, n))

    for i in range(iter):
        # V_pre is the estimate matrix
        V_pre = np.dot(W, H)
        # Error Matrix
        E = V - V_pre
        # error function
        err = np.sum(E * E)
        print('iter ' + str(i) + ': ' + str(err) + '.')

        # update matrix H
        Ha = np.dot(W.T, V)
        Hb = np.dot(W.T, np.dot(W, H))
        H[Hb!=0] = (H * Ha / Hb)[Hb!=0]

        # update matrix W
        Wc = np.dot(V, H.T)
        Wd = np.dot(W, np.dot(H, H.T))


        W[Wd!=0] = (W * Wc / Wd)[Wd!=0]

    return W, H
