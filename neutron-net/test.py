def plot(preds, labels, save_path, batch_size, error):
    # Weird bug was causing passed arrays to have indexing issues
    # The following is specific for the graphs generated in the paper graphics
    labels_1_a = labels[0][:,0]
    preds_1_a = preds[0][:,0]
    error_1_a = error[0][:,0]

    labels_1_b = labels[0][:,1]
    preds_1_b = preds[0][:,1]
    error_1_b = error[0][:,1]

    labels_2_a = labels[1][:,0]
    preds_2_a = preds[1][:,0]
    error_2_a = error[1][:,0]

    labels_2_b = labels[1][:,1]
    preds_2_b = preds[1][:,1]
    error_2_b = error[1][:,1]

    labels_2_c = labels[1][:,2]
    preds_2_c = preds[1][:,2]
    error_2_c = error[1][:,2]

    labels_2_d = labels[1][:,3]
    preds_2_d = preds[1][:,3]
    error_2_d = error[1][:,3]

    remainder = len(labels) % batch_size

    if remainder:
        labels = labels[:-remainder]
    
    total_plots = 6
    columns = 2
    rows = total_plots // columns 

    outer_grid = gridspec.GridSpec(3, 2, hspace=0.00, wspace=0.390, left=0.19, right=0.9, top=0.950, bottom=0.110)

    fig = plt.figure()
    pad = 55
    v_pad = 80

    ax0 = fig.add_subplot(outer_grid[0, 0])
    ax0.errorbar(labels_1_a, preds_1_a, error_1_a, fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax0.plot([0,1], [0,1], 'k', transform=ax0.transAxes)
    ax0.set_xlim([-250, 3250])
    ax0.set_ylim([-250, 3250])
    ax0.set_yticks([0, 1000, 2000, 3000])
    ax0.set_yticklabels([0, 1000, 2000, 3000])
    ax0.set_xticklabels([])
    ax0.annotate("$\mathregular{1^{i}}$", xy=(0., 0.5), xytext=(-ax0.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")

    ax0.set_facecolor("xkcd:very light blue")

    ax0_1 = fig.add_subplot(outer_grid[0, 1])
    ax0_1.errorbar(labels_1_b, preds_1_b, error_1_b,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax0_1.plot([0,1], [0,1], 'k', transform=ax0_1.transAxes)
    ax0_1.set_xlim([-0.1, 1.1])
    ax0_1.set_ylim([-0.1, 1.1])
    ax0_1.set_yticks([0, 0.5, 1])
    ax0_1.set_yticklabels([0, 0.5, 1])
    ax0_1.set_xticklabels([])
    ax0_1.set_facecolor("xkcd:very light blue")

    ax1 = fig.add_subplot(outer_grid[1, 0])
    ax1.errorbar(labels_2_a, preds_2_a, error_2_a,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax1.plot([0,1], [0,1], 'k', transform=ax1.transAxes)
    ax1.set_xlim([-250, 3250])
    ax1.set_ylim([-250, 3250])
    ax1.set_yticks([0, 1000, 2000, 3000])
    ax1.set_yticklabels([0, 1000, 2000, 3000])
    ax1.set_xticklabels([])
    ax1.set_ylabel("$\mathregular{Depth_{predict}\ (Å)}$", fontsize=11, weight="bold")
    ax1.annotate("$\mathregular{2^{i}}$", xy=(0, 0.5), xytext=(-ax0.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")

    ax2 = fig.add_subplot(outer_grid[1, 1])
    ax2.errorbar(labels_2_b, preds_2_b, error_2_b,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax2.plot([0,1], [0,1], 'k', transform=ax2.transAxes)
    ax2.set_xlim([-0.1, 1.1])
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels([0, 0.5, 1])
    ax2.set_xticklabels([])
    ax2.set_ylabel("$\mathregular{SLD_{predict}\ (fm\ Å^{-3})}$", fontsize=11, weight="bold")

    ax3 = fig.add_subplot(outer_grid[2, 0])
    ax3.errorbar(labels_2_c, preds_2_c, error_2_c,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="b",zorder=-130,markersize=4)
    ax3.plot([0,1], [0,1], 'k', transform=ax3.transAxes)
    ax3.set_xlim([-250, 3250])
    ax3.set_ylim([-250, 3250])
    ax3.set_yticks([0, 1000, 2000, 3000])
    ax3.set_yticklabels([0, 1000, 2000, 3000])
    ax3.set_xlabel("$\mathregular{Depth_{true}\ (Å)}$", fontsize=10, weight="bold")
    ax3.annotate("$\mathregular{2^{ii}}$", xy=(0, 0.5), xytext=(-ax3.yaxis.labelpad - pad, v_pad),
                        xycoords="axes points", textcoords="offset points",
                        size="large", ha="right", va="center")

    ax4 = fig.add_subplot(outer_grid[2, 1])
    ax4.errorbar(labels_2_d, preds_2_d, error_2_d,fmt="o",mec="k",mew=.5,alpha=.6,capsize=3,color="g",zorder=-130,markersize=4)
    ax4.plot([0,1], [0,1], 'k', transform=ax4.transAxes)
    ax4.set_xlim([-0.1, 1.1])
    ax4.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax4.set_xticklabels([0, 0.25, 0.5, 0.75, 1])
    ax4.set_ylim([-0.1, 1.1])
    ax4.set_yticks([0, 0.5, 1])
    ax4.set_yticklabels([0, 0.5, 1])
    ax4.set_xlabel("$\mathregular{SLD_{true}\ (fm\ Å^{-3})}$", fontsize=10, weight="bold")

    plt.savefig('onetwolayer.png', dpi=600)
