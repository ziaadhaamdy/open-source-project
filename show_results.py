import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_results(original, gray, glcm_f, dct_f, dwt_f, lbp_f):
    """Visualizes all algorithm results in an organized dark-themed figure."""

    fig = plt.figure(figsize=(20, 11), facecolor='#1a1a2e')
    fig.suptitle('Image Feature Extraction Results', color='white', fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.35)

    def dark_ax(ax, title=''):
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        if title:
            ax.set_title(title, color='white', fontsize=9, pad=4)

    # ── Row 0: Originals ──────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(original)
    dark_ax(ax0, 'Original RGB')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(gray, cmap='gray')
    dark_ax(ax1, 'Grayscale Input')
    ax1.axis('off')

    # ── GLCM ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(glcm_f['matrix'], cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    dark_ax(ax2, 'GLCM Matrix')

    ax3 = fig.add_subplot(gs[0, 3])
    names = ['Contrast', 'Corr.', 'Energy', 'Homog.']
    vals  = [glcm_f['contrast'], glcm_f['correlation'],
             glcm_f['energy'],   glcm_f['homogeneity']]
    bars = ax3.bar(names, vals, color=['#4fc3f7','#81c784','#ffb74d','#e57373'])
    dark_ax(ax3, 'GLCM Feature Values')
    ax3.tick_params(axis='x', labelrotation=15, labelsize=7, colors='white')
    ax3.tick_params(axis='y', colors='white')

    ax4 = fig.add_subplot(gs[0, 4])
    ax4.axis('off')
    dark_ax(ax4, 'GLCM Values')
    lines = [
        (f"Contrast:    {glcm_f['contrast']:.4f}",    '#4fc3f7'),
        (f"Correlation: {glcm_f['correlation']:.4f}", '#81c784'),
        (f"Energy:      {glcm_f['energy']:.4f}",      '#ffb74d'),
        (f"Homogeneity: {glcm_f['homogeneity']:.4f}", '#e57373'),
    ]
    for i, (txt, col) in enumerate(lines):
        ax4.text(0.05, 0.75 - i * 0.2, txt, color=col, fontsize=9,
                 transform=ax4.transAxes, family='monospace')

    # ── DCT ──────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0])
    dct_log = np.log(np.abs(dct_f['dct_matrix']) + 1)
    im2 = ax5.imshow(dct_log, cmap='jet', aspect='auto')
    plt.colorbar(im2, ax=ax5, fraction=0.046)
    dark_ax(ax5, 'DCT Spectrum (log)')

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(dct_f['reconstructed'], cmap='gray')
    dark_ax(ax6, 'DCT Reconstructed (Low Freq)')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    te = dct_f['total_energy']
    sizes = [dct_f['low_freq_energy'], dct_f['mid_freq_energy'], dct_f['high_freq_energy']]
    sizes_pct = [100 * s / te for s in sizes] if te > 0 else [33, 33, 34]
    ax7.pie(sizes_pct, labels=['Low', 'Mid', 'High'],
            colors=['#4fc3f7', '#ffb74d', '#e57373'],
            autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 8})
    dark_ax(ax7, 'DCT Energy Distribution')

    # ── DWT ──────────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[1, 3])
    LL1 = dwt_f['LL1']; LH1 = dwt_f['LH1']
    HL1 = dwt_f['HL1']; HH1 = dwt_f['HH1']

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    r2, c2 = LL1.shape
    rows, cols = gray.shape
    wdisp = np.zeros((rows, cols))
    wdisp[:r2, :c2]       = norm(LL1)
    wdisp[:r2, c2:]       = norm(np.abs(LH1[:, :cols-c2]))
    wdisp[r2:, :c2]       = norm(np.abs(HL1[:rows-r2, :]))
    wdisp[r2:, c2:]       = norm(np.abs(HH1[:rows-r2, :cols-c2]))
    ax8.imshow(wdisp, cmap='gray')
    dark_ax(ax8, 'DWT Level-1 Subbands')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[1, 4])
    x = np.arange(4)
    w = 0.25
    e1 = dwt_f['energy_L1']; e2 = dwt_f['energy_L2']; e3 = dwt_f['energy_L3']
    ax9.bar(x - w, e1, w, label='L1', color='#4fc3f7')
    ax9.bar(x,     e2, w, label='L2', color='#81c784')
    ax9.bar(x + w, e3, w, label='L3', color='#ffb74d')
    ax9.set_xticks(x)
    ax9.set_xticklabels(['LL','LH','HL','HH'], color='white', fontsize=8)
    ax9.legend(labelcolor='white', facecolor='#1a1a2e', fontsize=7)
    dark_ax(ax9, 'DWT Subband Energies')

    # ── LBP ──────────────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[2, 0])
    ax10.imshow(lbp_f['lbp_map'], cmap='gray')
    dark_ax(ax10, 'LBP Map')
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[2, 1])
    ax11.plot(lbp_f['histogram'], color='#4fc3f7', linewidth=1.2)
    ax11.fill_between(range(256), lbp_f['histogram'], alpha=0.3, color='#4fc3f7')
    dark_ax(ax11, 'LBP Histogram')
    ax11.set_xlabel('Pattern (0–255)', color='white', fontsize=8)
    ax11.set_ylabel('Frequency', color='white', fontsize=8)

    ax12 = fig.add_subplot(gs[2, 2])
    lbp_names = ['Uniformity', 'Entropy', 'Mean/10', 'Std/10']
    lbp_vals  = [lbp_f['uniformity'], lbp_f['entropy'],
                 lbp_f['mean_pattern'] / 10, lbp_f['std_pattern'] / 10]
    ax12.barh(lbp_names, lbp_vals, color=['#ffb74d','#4fc3f7','#81c784','#e57373'])
    dark_ax(ax12, 'LBP Descriptors')
    ax12.tick_params(axis='y', colors='white', labelsize=8)

    ax13 = fig.add_subplot(gs[2, 3])
    ax13.axis('off')
    dark_ax(ax13, 'LBP Values')
    lbp_lines = [
        (f"Uniformity:  {lbp_f['uniformity']:.4f}",   '#ffb74d'),
        (f"Entropy:     {lbp_f['entropy']:.4f} bits",  '#4fc3f7'),
        (f"Mean:        {lbp_f['mean_pattern']:.2f}",  '#81c784'),
        (f"Std:         {lbp_f['std_pattern']:.2f}",   '#e57373'),
    ]
    for i, (txt, col) in enumerate(lbp_lines):
        ax13.text(0.05, 0.78 - i * 0.2, txt, color=col, fontsize=9,
                  transform=ax13.transAxes, family='monospace')

    plt.savefig('results_all.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.show()
    print("\n✅ Full results saved to: results_all.png")
