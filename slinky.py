import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def slinky_plot(M,filename=None,dpi=70,class_labels=None,time_labels=None,class_label_fontsize=11,time_label_fontsize=14,
            count_fontsize=11,plot_arrows=True,legend=False,additional_padding=1.0,min_count_for_label=1,
            pad=1/6,scale_figsize=1.0,vertical_margin_counts=True,color_palette=None,legend_loc='best',
            legend_fontsize=11,scale_transparency=False,min_cell_dim=0):
    """
        Plots a slinky plot given a sequence of transition matrices.
    
        Inputs:
            M : array-like of shape (n_time_points-1,n_classes,n_classes)
                Series of transition matrices.
            filename : string, default=None
                Filename to save plot.
            dpi : int, default=70
                DPI for saved image if saving as non-vector format.
            class_labels : array-like of shape (n_classes,), default=None
                List of class labels to be printed on the margins of the plot. If None, 
                default values of ['c1', 'c2',...] will be used.
            time_labels : array-like of shape (n_time_points,), default=None
                List of timestep labels to be printed on the margins of the plot. If None,
                default values of ['t0','t1','t2',...] will be used.
            class_label_fontsize : int, defulat=11
                Font size for class labels.
            time_label_fontsize : int, default=14
                Font size for time labels.
            count_fontsize : int, default=11
                Font size for the counts shown in the plot.
            plot_arrows : bool, default=True
                If True, guidline arrows will be plotted.
            legend : bool, default=False
                If True, a legend will be plotted.
            additional_padding : float, default=1.0
                The amount of extra padding to add around the plot. Use this expand the plot 
                if the labels are getting cut off.
            min_count_for_label : int, default=1
                If the count for a particular class on a particular margin is less than this 
                number, the label and counts for that class will not be printed along this margin.
                This prevents labels from overrunning box boundaries. The default is to supress 
                labels only if the count is zero.
            pad : float, default = 1/6
                Controls the padding added to text labels. Given as a proportion of the
                transition matrix side length.
            scale_figsize : float, default=1.0
                Can be used to scale up or down the figure size.
            vertical_margin_counts : bool, default=True
                If True, column counts are rotated 90 degrees.
            color_palette : list-like of colors, default=None
                Color cycle for the different classes. If none, the seaborn "colorblind" 
                palette is used.
            legend_loc : str or tuple of floats, default='best'
                Location argument for matplotlib.pyplot.legend.
            legend_fontsize : int, default=11
                Fontsize for the legend.
            scale_transparency : bool, default=False
                of the corresponding margins is under 100.
                If the scale_transparency argument is set to True, then the transparency 
                of a cell is set to the outgoing proportion for that cell. For example, 
                if the cell corresponding to  the number moving from class 1 at time 1 to 
                class 2 at time 2 is 50 and the class 2 total at time 2 is 100, then the 
                transparency for that cell is set to 50%.
            min_cell_dim : int, default=0
                The min_cell_dim argument allows you to specify a minimum size for any of 
                the margins. For example, if min_cell_dim=10, then no margin will have a 
                size smaller than 10/N (where N is the total count). The displayed counts 
                within each cell and on the margins remain unchanged. Additionally, margins 
                with counts larger than min_cell_dim are scaled down to accommodate minimum 
                sized margins while maintaining square transition matrices.
                
    """
            
    M = M.astype(int)        
    
    N = M[0].sum()
    T = M.shape[0]
    C = M.shape[1]
    
    assert M.shape[1] == M.shape[2], f"M must have shape {n_time_points-1,n_classes,n_classes}. M.shape[1] != M.shape[0]."
    assert np.all(M.sum((1,2)) == N), f"All transition matrices must sum to the same number. Input sums: {M.sum((1,2))}"
    for t in range(T-1):
        assert np.all(M[t].sum(0) == M[t+1].sum(1)), f"The corresponding margins of consecutive transition matrices must be equal. The margins for time t={t+1} are: {M[t].sum(0)} != {M[t+1].sum(1)}"
        
    if class_labels is not None:
        assert len(class_labels)==C, f"class_labels must be a length {C} list"
        for cl in class_labels:
            try:
                tmp = str(cl)
            except:
                raise ValueError("All values in class_labels must be printable.")
    else:
        class_labels = [f'c{c+1}' for c in range(C)]
                
    if time_labels is not None:
        assert len(time_labels)==T+1, f"class_labels must be a length {T+1} list"
        for tl in time_labels:
            try:
                tmp = str(tl)
            except:
                raise ValueError("All values in time_labels must be printable.")
    else:
        time_labels = [f't{t}' for t in range(T+1)]
        
    if color_palette is None:
        cp = sns.color_palette("colorblind")
    else:
        cp = color_palette

    additional_padding = additional_padding*N
    pad = N*pad
    n_mats_wide = np.floor(T/2) + 1
    fs = (scale_figsize*4*n_mats_wide,scale_figsize*4*n_mats_wide)
    fig = plt.figure(figsize=fs)

    xmin = -additional_padding
    xmax = n_mats_wide*N + (n_mats_wide-1)*pad + N
    ymax = additional_padding
    ymin = -xmax
    # plt.plot([xmin,xmax],[ymin,ymax])
    #  = max(xmax-xmin,ymax-ymin)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))

    ax = plt.gca()
    ax.set_aspect(1)
    x = 0
    y = -N
    for t in range(T):
    
        rec = patches.Rectangle((x,y),N,N,ec='k',fc='w')
        ax.add_patch(rec)
    

        Mt = M[t]
    
        if t%2 == 0:
            Mt = M[t]
        else:
            Mt = M[t].T

        raw_x_margins = Mt.sum(1)
        raw_y_margins = Mt.sum(0)
        
        plot_x_margins = np.zeros(raw_x_margins.shape)
        plot_y_margins = np.zeros(raw_y_margins.shape)
        
        plot_remainder = N
        count_remainder = N
        idxs = np.argsort(raw_x_margins)
        sorted_margins = np.sort(raw_x_margins)
        for i,xm in zip(idxs,sorted_margins):
            if xm == 0:
                continue
            plot_x_margins[i] = max(min_cell_dim,xm*plot_remainder/count_remainder)
            count_remainder -= xm
            plot_remainder -= plot_x_margins[i]
            
        plot_remainder = N
        count_remainder = N
        idxs = np.argsort(raw_y_margins)
        sorted_margins = np.sort(raw_y_margins)
        for i,ym in zip(idxs,sorted_margins):
            if ym == 0:
                continue
            plot_y_margins[i] = max(min_cell_dim,ym*plot_remainder/count_remainder)
            count_remainder -= ym
            plot_remainder -= plot_y_margins[i]
            
        # add the margins first
        if t==0:
            d = 0
            for c in range(C):
                x_text = x + d + plot_x_margins[c]/2.0
                y_text = y + N + pad/2
                if raw_x_margins[c] >= min_count_for_label:
                    plt.text(x_text, y_text, f'{raw_x_margins[c]:,}',
                             verticalalignment='center',
                             horizontalalignment='center',
                             fontsize=count_fontsize,
                             rotation=90*vertical_margin_counts)
            
            
                rec = patches.Rectangle((x+d,y+N),plot_x_margins[c],pad,ec=None,fc=cp[c])
                ax.add_patch(rec)
            
                d += plot_x_margins[c]
        d = 0
        for c in range(C):
            if t%2 == 0:
                y_text = y + d + plot_y_margins[c]/2.0
                x_text = x + N + pad/2
                if raw_y_margins[c] >= min_count_for_label:
                    plt.text(x_text, y_text, f'{raw_y_margins[c]:,}',
                             verticalalignment='center',
                             horizontalalignment='center',
                             fontsize=count_fontsize)
            
                rec = patches.Rectangle((x+N,y+d),pad,plot_y_margins[c],ec=None,fc=cp[c])
                ax.add_patch(rec)
            
                d += plot_y_margins[c]
            else:
                x_text = x + d + plot_x_margins[c]/2.0
                y_text = y - pad/2
                if raw_x_margins[c] >= min_count_for_label:
                    plt.text(x_text, y_text, f'{raw_x_margins[c]:,}',
                             verticalalignment='center',
                             horizontalalignment='center',
                             fontsize=count_fontsize,
                             rotation=90*vertical_margin_counts)
            
            
                rec = patches.Rectangle((x+d,y-pad),plot_x_margins[c],pad,ec=None,fc=cp[c])
                ax.add_patch(rec)
            
                d += plot_x_margins[c]
    
        # Add counts to the boxes
        xd = 0
        for c1 in range(C):
            yd = 0
            for c2 in range(C):
            
                # plot box for this transition
                if t%2 == 0:
                    fc = cp[c2]
                    if scale_transparency and raw_y_margins[c2] > 0:
                        alpha = Mt[c1,c2]/raw_y_margins[c2]
                    else:
                        alpha = 1.0
                else:
                    fc = cp[c1]
                    if scale_transparency and raw_x_margins[c1] > 0:
                        alpha = Mt[c1,c2]/raw_x_margins[c1]
                    else:
                        alpha = 1.0
                    
                rec = patches.Rectangle((x+xd,y+yd),plot_x_margins[c1],plot_y_margins[c2],ec='k',fc=fc+(alpha,),zorder=2)
                ax.add_patch(rec)
            
                # add count to center of transition box
                x_text = x + xd + plot_x_margins[c1]/2.0
                y_text = y + yd + plot_y_margins[c2]/2.0
                if raw_x_margins[c1] >= min_count_for_label and raw_y_margins[c2] >= min_count_for_label:
                    
                    plt.text(x_text, y_text, f'{Mt[c1,c2]:,}',
                             verticalalignment='center',
                             horizontalalignment='center',
                             fontsize=count_fontsize)
            
                yd += plot_y_margins[c2]
        
            xd += plot_x_margins[c1]
        
        # add class labels
        if t == 0:
            d = 0
            y_top = -np.inf
            for c in range(C):
                x_text = x + d + plot_x_margins[c]/2
                y_text = y + N + pad + N/20
                r = 'vertical'
                va = 'bottom'
                ha = 'center'
                if raw_x_margins[c] >= min_count_for_label:
                    text = plt.text(x_text,y_text,class_labels[c],
                             verticalalignment=va,
                             horizontalalignment=ha,
                             rotation=r,
                             fontsize=class_label_fontsize)

                    plt.gcf().canvas.draw()


                    # get bounding box of the text 
                    # in the units of the data
                    bbox = text.get_window_extent()\
                        .transformed(plt.gca().transData.inverted())
            
                    y_top = max(y_top,bbox.y1 + N/20)

                d += plot_x_margins[c]
        
            d = 0
            y_bot = y+N+pad
            for c in range(C):
                rec = patches.Rectangle((x+d,y_bot),plot_x_margins[c],y_top-y_bot,ec=None,fc=cp[c])
                ax.add_patch(rec)
                d += plot_x_margins[c]
            
            plt.text(x + N/2,y_top + N/20,time_labels[0],
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     fontsize=time_label_fontsize)
        d = 0
        y_top = -np.inf
        x_left = np.inf
        for c in range(C):
            if t%2 == 1:
                x_text = x + d + plot_x_margins[c]/2
                y_text = y + N + N/20
                r = 'vertical'
                va = 'bottom'
                ha = 'center'
                cl = class_labels[c]
                d += plot_x_margins[c]
                print_text = raw_x_margins[c] >= min_count_for_label
            else:
                x_text = x - N/20
                y_text = y + d + plot_y_margins[c]/2
                r = 'horizontal'
                va = 'center'
                ha = 'right'
                cl = class_labels[c]
                d += plot_y_margins[c]
                print_text = raw_y_margins[c] >= min_count_for_label

            if print_text:
                text=plt.text(x_text,y_text,cl,
                         verticalalignment=va,
                         horizontalalignment=ha,
                         rotation=r,
                         fontsize=class_label_fontsize)
        
                plt.gcf().canvas.draw()


                # get bounding box of the text 
                # in the units of the data
                bbox = text.get_window_extent()\
                    .transformed(plt.gca().transData.inverted())
        
                y_top = max(y_top,bbox.y1 + N/20)
                x_left = min(x_left,bbox.x0 - N/20)
            
        # Add label
        if t%2 == 0:
            d = 0
            for c in range(C):
                rec = patches.Rectangle((x_left,y + d),x-x_left,plot_y_margins[c],ec=None,fc=cp[c])
                ax.add_patch(rec)
                d += plot_y_margins[c]
            
            plt.text(x_left - N/20,y + N/2,time_labels[t+1],
                     verticalalignment='center',
                     horizontalalignment='right',
                     fontsize=time_label_fontsize)
        else:
            d = 0
            y_bot = y+N
            for c in range(C):
                rec = patches.Rectangle((x+d,y_bot),plot_x_margins[c],y_top-y_bot,ec=None,fc=cp[c])
                ax.add_patch(rec)
                d += plot_x_margins[c]
            
            plt.text(x + N/2,y_top + N/20,time_labels[t+1],
                     verticalalignment='bottom',
                     horizontalalignment='center',
                     fontsize=time_label_fontsize)
        
        
        # Add arrows
        if plot_arrows:
            if t%2 == 0:
                arrow_start = (x_left-N/6, y+N/6)
                arrow_end = (x_left+N/6, y-N/6)
                cs = "arc3,rad=.5"
            else:
                arrow_start = (x+N-N/6,y_top+N/6)
                arrow_end = (x+N+N/6,y_top-N/6)
                cs = "arc3,rad=-.5"

            style = "Simple, tail_width=0.5, head_width=4, head_length=8"
            kw = dict(arrowstyle=style, color="k")

            arrow = patches.FancyArrowPatch(arrow_start, arrow_end,
                                         connectionstyle=cs, **kw)
                                         
            ax.add_patch(arrow)
        
        # increment x and y
        if t%2 == 0:
            x += N + pad
        else:
            y -= N + pad
        
    if legend:
        les = []
        for c in range(C):
            les.append(patches.Patch(ec=None,fc=cp[c],label=class_labels[c]))
        ax.legend(handles=les, loc=legend_loc,fontsize=legend_fontsize)
        
    plt.axis('off')
    plt.tight_layout(pad=0.25)
    if filename is not None:
        plt.savefig(filename,dpi=dpi, bbox_inches='tight',pad_inches=0.25)
    # plt.show()