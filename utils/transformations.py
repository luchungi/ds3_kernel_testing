import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

def leadlag(X):
    '''
    Returns lead-lag-transformed stream of X

    Arguments:
        X: list, whose elements are tuples of the form
        (time, value).

    Returns:
        list of points on the plane, the lead-lag
        transformed stream of X
    '''

    l=[]

    for j in range(2*(len(X))-1):
        i1=j//2
        i2=j//2
        if j%2!=0:
            i1+=1
        l.append((X[i1][1], X[i2][1]))

    return l

def plotLeadLag(X, diagonal=True):
    '''
    Plots the lead-laged transformed path X. If diagonal
    is True, a line joining the start and ending points
    is displayed.

    Arguments:
        X: list, whose elements are tuples of the form
        (X^lead, X^lag) diagonal: boolean, default is
        True. If True, a line joining the start and
        ending points is displayed.
    '''
    for i in range(len(X)-1):
        plt.plot([X[i][1], X[i+1][1]], [X[i][0], X[i+1][0]],
                    color='k', linestyle='-', linewidth=2)


    # Show the diagonal, if diagonal is true
    if diagonal:
        plt.plot([min(min([p[0] for p in X]), min([p[1]
                for p in X])), max(max([p[0] for p in X]),
                max([p[1] for p in X]))], [min(min([p[0]
                for p in X]), min([p[1] for p in X])),
                max(max([p[0] for p in X]), max([p[1] for
                p in X]))], color='#BDBDBD', linestyle='-',
                linewidth=1)

    axes=plt.gca()
    axes.set_xlim([min([p[1] for p in X])-1, max([p[1] for
                  p in X])+1])
    axes.set_ylim([min([p[0] for p in X])-1, max([p[0] for
                  p in X])+1])
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    axes.set_aspect('equal', 'datalim')
    plt.show()

def timejoined(X):
    '''
    Returns time-joined transformation of the stream of
    data X

    Arguments:
        X: list, whose elements are tuples of the form
        (time, value).

    Returns:
        list of points on the plane, the time-joined
        transformed stream of X
    '''
    X.append(X[-1])
    l=[]

    for j in range(2*(len(X))+1+2):
            if j==0:
                    l.append((X[j][0], 0))
                    continue
            for i in range(len(X)-1):
                    if j==2*i+1:
                            l.append((X[i][0], X[i][1]))
                            break
                    if j==2*i+2:
                            l.append((X[i+1][0], X[i][1]))
                            break
    return l

def plottimejoined(X):
    '''
    Plots the time-joined transfomed path X.

    Arguments:
        X: list, whose elements are tuples of the form (t, X)
    '''

    for i in range(len(X)-1):
        plt.plot([X[i][0], X[i+1][0]], [X[i][1], X[i+1][1]],
                color='k', linestyle='-', linewidth=2)

    axes=plt.gca()
    axes.set_xlim([min([p[0] for p in X]), max([p[0] for p in X])+1])
    axes.set_ylim([min([p[1] for p in X]), max([p[1] for p in X])+1])
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.get_xaxis().get_major_formatter().set_useOffset(False)
    # axes.set_aspect('equal', 'datalim')
    plt.show()

def lead_lag_transformation(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    features = x.shape[-1]
    factor = 2*features
    stack_list = []
    x = np.repeat(x.T, factor, axis=1) # transpose required
    for i in range(factor):
        stack_list.append(np.roll(x[i%features,...], i)[factor-1:])

    x = np.stack(stack_list, axis=-1)
    if x.ndim == 3:
        x = x.transpose(1, 0, 2)
    return x