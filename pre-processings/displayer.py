################################################################################
import numpy            as np
import matplotlib, sys
import logging

## to avoid bug when it is run without graphic interfaces
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except ImportError:
    # print ('Warning importing GTK3Agg: ', sys.exc_info()[0])
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


################################################################################
def display_matrix (t, f, mat, path_save, bandwidth = None):
################################################################################
    # from: https://matplotlib.org/examples/pylab_examples/scatter_hist.html

    # definitions for the axes
    left, bottom  = 0.1, 0.1
    height, width = 0.5, 0.5

    size_hist = 0.2
    colorbar_size = 0.05

    spacing = 0.005
    spacing_colorbar = 0.05


    rect_colorbar = [left, bottom, colorbar_size, height]
    rect_scatter  = [left + colorbar_size + spacing_colorbar, bottom, width, height]

    rect_histx    = [left + colorbar_size + spacing_colorbar, bottom + height + spacing, width, size_hist]
    rect_histy    = [left + colorbar_size + spacing_colorbar + width + spacing, bottom, size_hist, height]


    # start with a rectangular Figure
    fig = plt.figure (figsize = (16, 9))

    ax_scatter = plt.axes (rect_scatter)
    ax_scatter.tick_params (direction='in', top=True, right=True)

    ax_histx = plt.axes (rect_histx)
    ax_histx.tick_params (direction='in', labelbottom=False)

    ax_histy = plt.axes (rect_histy)
    ax_histy.tick_params (direction='in', labelleft=False)

    ax_colorbar = plt.axes (rect_colorbar)
    ax_colorbar.tick_params (direction='in', labelright=False)



    im = ax_scatter.imshow (mat, cmap = 'Blues', interpolation ='none', aspect='auto',
                            origin ='lower',
                            extent = [t.min (), t.max (), f.min (), f.max ()])

    plt.colorbar(im, cax = ax_colorbar)
    ax_colorbar.yaxis.set_ticks_position('left')

    ax_scatter.set_xlabel ('Time (s)')
    ax_scatter.set_ylabel ('Freq (MHz)')

    ax_histx.plot (t, mat.mean (0))
    ax_histx.set_ylabel ('NICV')

    ax_histy.plot (mat.mean (1), f)
    ax_histy.set_xlabel ('NICV')

    ax_histx.set_xlim(ax_scatter.get_xlim ())
    ax_histy.set_ylim(ax_scatter.get_ylim ())

    # projection of axis
    if (bandwidth is not None):
        i = 0
        while (i < len (bandwidth) - 1):
            j = i
            while (bandwidth [j] - bandwidth [j + 1] == 1):
                j += 1
                ax_histy.axhspan (f [bandwidth [i]], f [bandwidth [j]], color = 'red')

            if (i == j):
                ax_histy.axhspan (f [bandwidth [i]], f [bandwidth [i]], color = 'red')
                j +=1

            i = j
        # ax_histy.legend ()

    ## save if a file is provided
    if (path_save == None):
        plt.show ()
    ## otherwise save it
    else:
        plt.savefig (f'{path_save}')
        plt.close ()
