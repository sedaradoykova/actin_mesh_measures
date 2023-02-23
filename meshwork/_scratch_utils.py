import scipy.ndimage, cv2, os
import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from matplotlib_scalebar.scalebar import ScaleBar
from actin_meshwork_analysis.meshwork.actinimg import get_ActinImg

def plt_threshold_diagnostic(actimg, linprof_original):
    linprof = linprof_original[np.argwhere(linprof_original>0)]
    percentiles = list(map(lambda percent: np.percentile(linprof, percent), [90,95,98]))

    max_lprof = np.max(linprof_original)
    plt_ceil = 1.33*max_lprof
    while plt_ceil < 10:
        plt_ceil *= 10
    plt_ceil = int(np.ceil(plt_ceil)*10)
    factor = int(np.ceil(plt_ceil / (1.2*max_lprof)))


    coords = _line_profile_coordinates((0,0), actimg.shape)
    order, mode, cval = 1, 'reflect', 0.0
    pixels = scipy.ndimage.map_coordinates(actimg.manipulated_stack, coords, prefilter=order>1, 
                                           order=order, mode=mode, cval=cval)
    pixels[pixels <= 1e-10] = 0
    upsized_im = cv2.resize(np.transpose(pixels), (pixels.shape[0], 50))


    fig = plt.figure(figsize=(8, 7))
    fig.subplots_adjust(hspace=0.02)
    fig.add_subplot(221)
    # plt.subplot(2,2,1)
    plt.imshow(actimg.manipulated_stack,cmap='gray')
    scalebar = ScaleBar(actimg.resolution, 'nm', box_color='None', color='black', location='upper left') 
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    plt.plot((0,actimg.shape[0]),(0,actimg.shape[1]), color='black')
    plt.title('(A) Minimum projection of\nsteerable Gaussian filter response')
    # ax = plt.subplot(2,2,3)
    ax = fig.add_subplot(223)
    plt.imshow(upsized_im, cmap='gray', extent=[0, upsized_im.shape[1], plt_ceil-50, plt_ceil])
    plt.plot(linprof_original*factor, color='black')
    plt.ylim(0,plt_ceil)
    oticks = ax.get_yticks()
    nlabels = np.round(np.linspace(0,max_lprof,len(oticks[:-1])), 4)
    plt.yticks(ticks=oticks[:-1], labels=nlabels)
    plt.xlabel('Pixel location on line')
    plt.ylabel('Pixel intensity value')
    plt.title('(B) Line profile of steerable\nGaussian filter response')
    # plt.subplot(2,2,2)
    ax = fig.add_subplot(122)
    plt.hist(linprof,color='#A8A8A8')
    xlims, ymax= ax.get_xlim(), ax.get_ylim()[1]
    plt.vlines(x=percentiles, ymin=0, ymax=ymax, colors='black', ls='--', lw=2, label='90,95,98 percentile')
    for x in percentiles:
        plt.text(x,0.9*ymax,np.format_float_scientific(x,2))
    plt.xlabel('Pixel intensity value')
    plt.ylabel('Count')
    plt.title('(C) Histogram of\nnon-negative intensity values')
    plt.legend()
    #plt.tight_layout()
    plt.show();

if '__name__' == '__main__':
    plt_threshold_diagnostic(actimg, linprof_original)

    # choosing n bins 
    int(np.subtract(np.max(linprof),np.min(linprof)) / ( 2.59*np.subtract(*np.percentile(linprof, [75, 25]))/np.cbrt(len(linprof)) ))

    plt.hist(linprof,color='#A8A8A8',bins=10) # 154 Scott's normal reference (2.59*IQR) or 200 for Freedman–Diaconis rule (2*IQR) 
    plt.show()
    2*np.cbrt(len(linprof))

    np.subtract(np.max(linprof),np.min(linprof))*np.cbrt(len(linprof))/(3.49*np.std(linprof))


    np.format_float_scientific(0.002341088804324921,2)
    list(map(lambda percent: np.percentile(linprof, percent), [90,95,98]))

    data_path = os.path.join(os.getcwd(), "actin_meshwork_analysis/process_data/sample_data/CARs")
    os.listdir(data_path)

    actimg = get_ActinImg('3min_FOV3_decon.tif', data_path) # base = [1,4], cyto = [4,7] 
    actimg.normalise()
    actimg.steerable_gauss_2order_thetas(thetas=[0,60,120],sigma=2,substack=[3,5],visualise=False)
    actimg.z_project_min()
    #actimg.visualise_stack('manipulated',colmap='gray')
    linprof_original = profile_line(actimg.manipulated_stack, (0,0), actimg.shape)
    linprof = linprof_original[np.argwhere(linprof_original>0)]

    plt_threshold_diagnostic(actimg, linprof_original)




def _line_profile_coordinates(src, dst, linewidth=1):
    """Skimage line_profile helper. 
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.stack([perp_rows, perp_cols])