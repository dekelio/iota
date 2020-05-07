"""
Code for visualizing IOTA results

"""

import os
import matplotlib
if "DISPLAY" not in os.environ:
    #raise ValueError('Gal: DISPLAY not in os.environ')
    matplotlib.use('Agg')
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from graphviz import Digraph
import numpy as np
import pandas as pd
import pickle
import utils
import seaborn as sns
from collections import Counter


# importing mcolors, requires updating matplotlib which may cause conflicts.
def set_colors():
    # See here: https://matplotlib.org/2.1.1/gallery/color/named_colors.html
    colors = dict()
    colors['skyblue'] = u'#87CEEB'
    colors['steelblue'] = u'#4682B4'
    colors['maroon'] = u'#800000'
    colors['orange'] = u'#FFA500'
    colors['lightcoral'] = u'#F08080'
    colors['y'] = u'#B0B000'
    colors['red'] = u'#FF0000'
    colors['black'] = u'#000000'
    return colors


def get_models_style():
    # Color palate: steelblue, olivedrab, slateblue, chocolate, firebrick
    # colors = set_colors() # For earlier versions of matplotlib
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    metrics = dict()
    metrics[6] = {'name': 'cH', 'legend': '$cw-\Delta H$',
                  'color': colors['royalblue'], 'fmt': '-', 'marker': 's'}
    metrics[5] = {'name': 'cDKL', 'legend': '$cw-D_{KL}$',
                  'color': colors['steelblue'], 'fmt': '-', 'marker': '*'}
    metrics[3] = {'name': 'cMI', 'legend': '$cw-Image \Delta H$',
                  'color': colors['chocolate'], 'fmt': '-', 'marker': '.'}
    metrics[2] = {'name': 'cSingleton', 'legend': '$cw-Singleton$',
                  'color': colors['firebrick'], 'fmt': '-', 'marker': 'x'}
    metrics[4] = {'name': 'cPX', 'legend': '$cw-p(x)$',
                  'color': colors['y'], 'fmt': '-', 'marker': '^'}
    metrics[1] = {'name': 'Confidence', 'legend': '$confidence$',
                  'color': colors['olivedrab'], 'fmt': '-', 'marker': '+'}
    metrics[0] = {'name': 'random', 'legend': '$random$',
                  'color': colors['grey'], 'fmt': '-', 'marker': ''}
    # metrics[7] = {'name': 'H', 'legend': 'H',
    #               'color': colors['red'], 'fmt': '--', 'marker': ''}
    # metrics[8] = {'name': 'mi', 'legend': '$Image \Delta H$',
    #               'color': 'skyblue', 'fmt': '--', 'marker': ''}
    # metrics[9] = {'name': 'px', 'legend': '$p(x)$',
    #               'color': colors['steelblue'], 'fmt': '--', 'marker':''}# TFIDF
    # metrics[10] = {'name': 'singleton', 'legend': '$Singleton$',
    #               'color': colors['maroon'], 'fmt': '--', 'marker': ''}
    # metrics[11] = {'name': 'dkl', 'legend': '$D_{KL}$',
    #               'color': colors['y'], 'fmt': '--', 'marker':''}
    # metrics[12] = {'name': 'imd_H', 'legend': '$cond_H$',
    #               'color': colors['purple'], 'fmt': '.-', 'marker':''}
    # metrics[13] = {'name': 'imd_dkl', 'legend': '$cond_dkl$',
    #               'color': colors['orange'], 'fmt': '.-', 'marker':''}
    # metrics[14] = {'name': 'imd_cH', 'legend': '$cond_cH$',
    #                'color': colors['blue'], 'fmt': '.-', 'marker': ''}
    # metrics[15] = {'name': 'imd_cDKL', 'legend': '$cond_cDKL$',
    #                'color': colors['cyan'], 'fmt': '.-', 'marker': ''}
    # fs = {'axis': 26, 'ticks': 16, 'font_size': 18, 'legend': 18}
    fs = {'axis': 24, 'ticks': 20, 'font_size': 18, 'legend': 18}
    line_width = 2
    # marker_size = 7
    marker_size = 10
    return metrics, fs, line_width, marker_size


def save_figure_and_close(files, param_string, filename, extension='.png'):
    path = files['results_dir'] + param_string
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = path + '/' + param_string + '_' + filename + extension
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    print('Write figure to [%s]' % utils.blue(filepath))
    plt.close()


def plot_correlation(hp, files, df, headers=['Confidence', 'cH', 'cMI',
                                         'cSingleton',
                                        'H', 'px', 'random', 'mi', 'singleton',
                                        'cPX', 'cDKL', 'dkl'], show=False):
    add_eval_set_to_string = True
    param_string = utils.create_param_string(hp, add_eval_set_to_string)
    metrics, font_size, lw, ms = get_models_style()
    disp = dict()
    for n in metrics.values():
        disp[n['name']] = n['legend']
    cor_cols = df[headers]
    corr = cor_cols.corr(method='pearson')
    d = [disp[n] for n in corr.columns]
    sns.set(font_scale=1)
    sns.heatmap(corr, xticklabels=d, yticklabels=d, cmap='viridis')

    if show: plt.show()
    save_figure_and_close(files, param_string, 'correlation.png')


def print_top_pr(average_precision, average_recall):
    metrics, _, _, _ = get_models_style()
    for key in metrics:
        m = metrics[key]['name']
        p = average_precision[m].tolist()[0]
        r = average_recall[m].tolist()[0]
        print('%20s  p@1=%4.2f r@1=%4.2f' % (m, p[0], r[0]))


def plot_precision(hp, files, average_precision, err, raters_p, show=False):
    k = hp['k']
    metrics, fs, lw, ms = get_models_style()
    k_vec = range(1, k + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    max_p = 0
    for key in metrics:
        p = average_precision[metrics[key]['name']].tolist()[0]
        if max(p) > max_p: max_p = max(p)
        plt.errorbar(k_vec, p,
                     yerr=err[metrics[key]['name']],
                     linewidth=lw,
                     label=metrics[key]['legend'],
                     fmt=metrics[key]['fmt'],
                     color=metrics[key]['color'],
                     marker=metrics[key]['marker'], markersize=ms)
    ax.set_xlabel('Top-k', fontsize=fs['axis'])
    ax.set_ylabel('Precision', fontsize=fs['axis'])
    ax.set_xlim(left=1, right=k)
    ax.set_ylim(bottom=0, top=max_p+0.1)
    plt.rc('xtick', labelsize=fs['ticks'])
    plt.rc('ytick', labelsize=fs['ticks'])

    # Plot raters agreement
    plt.axhline(y=raters_p, color='grey', linestyle='--', linewidth=1,
                label='rater agreement')
    plt.legend(bbox_to_anchor=(-0.04, 1.02, 1, 0.2), loc="lower left",
               mode="expand", ncol=3, prop={'size': 14}, frameon=False,
               borderaxespad=0)
    if show: plt.show()

    add_eval_set_to_string = True
    param_string = utils.create_param_string(hp, add_eval_set_to_string)
    save_figure_and_close(files, param_string, 'precision.png')


def plot_recall(hp, files, average_recall, err, show=False):
    k = hp['k']
    add_eval_set_to_string = True
    param_string = utils.create_param_string(hp, add_eval_set_to_string)
    metrics, fs, lw, ms = get_models_style()
    k_vec = range(1, k + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    max_r = 0
    for key in metrics:
        r = average_recall[metrics[key]['name']].tolist()[0]
        if max(r) > max_r: max_r = max(r)
        plt.errorbar(k_vec, r, yerr=err[metrics[key]['name']], linewidth=lw,
                     label=metrics[key]['legend'], fmt=metrics[key]['fmt'],
                     color=metrics[key]['color'], marker=metrics[key]['marker'],
                     markersize=ms)
    plt.legend(bbox_to_anchor=(-0.04, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", ncol=3, prop={'size': 14},
                    frameon=False, borderaxespad=0)
    ax.set_xlabel('Top-k', fontsize=fs['axis'])
    ax.set_ylabel('Recall', fontsize=fs['axis'])
    ax.set_xlim(left=1, right=k)
    ax.set_ylim(bottom=0, top=max_r+0.1)
    plt.rc('xtick', labelsize=fs['ticks'])
    plt.rc('ytick', labelsize=fs['ticks'])
    if show: plt.show()
    save_figure_and_close(files, param_string, 'recall.png')


def plot_precision_vs_recall(hp, files, average_precision, average_recall,
                             raters_p, show=False, legend_loc='none'):
    add_eval_set_to_string = True
    param_string = utils.create_param_string(hp, add_eval_set_to_string)
    metrics, fs, lw, ms = get_models_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    # fig, ax = plt.subplots()
    max_pr = 0
    for key in metrics.keys()[::-1]:
        r = average_recall[metrics[key]['name']].tolist()[0]
        p = average_precision[metrics[key]['name']].tolist()[0]
        if max(p) > max_pr: max_pr = max(p)
        plt.plot(r, p, lw=lw, label=metrics[key]['legend'],
                 color=metrics[key][
                     'color'], marker=metrics[key]['marker'], mew=2, ms=ms)
    # plt.axis('equal')
    ax.set_xlabel('Recall', fontsize=fs['axis'])
    ax.set_ylabel('Precision', fontsize=fs['axis'])
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)  # max_pr+0.1)
    plt.rc('xtick', labelsize=fs['ticks'])
    plt.rc('ytick', labelsize=fs['ticks'])
    plt.axhline(y=raters_p, color='grey', linestyle='--', linewidth=1,
                label='rater agreement')
    plt.text(0.02, 0.685, 'Inter-rater agreement', fontsize=20)
    # plt.text(0.02, 0.65, 'Inter-rater agreement', fontsize=20)
    legend_fs = 18
    if legend_loc == 'up':
        plt.legend(bbox_to_anchor=(-0.04, 1.02, 1, 0.2), loc="lower left",
                   mode="expand", ncol=3,
                   prop={'size': legend_fs},
                   frameon=False, borderaxespad=0)
    if legend_loc == 'in':
        plt.legend(loc="upper left", ncol=2,
                   prop={'size': legend_fs, 'weight': 'bold'},
                   frameon=False, borderaxespad=0.9)
    if legend_loc == 'side':
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1,
                   prop={'size': legend_fs}, frameon=False, borderaxespad=0.9)
    if show: plt.show()
    fig.dpi = 600  # 200
    save_figure_and_close(files, param_string, 'pr', extension='.png')


def write_models_to_html(image_metrics, hp, files):
    image_metrics.sort_values(['ImageID', 'cH'], ascending=[True, False],
                              inplace=True)
    add_eval_set = True
    param_string_eval_set = utils.create_param_string(hp, add_eval_set)
    out_filename = ('%s/%s/%s.html' % (files['results_dir'],
                                       param_string_eval_set,
                                       param_string_eval_set))

    # Map metric to its display name.
    metrics, metrics_display = [], dict()
    metrics_style, fs, line_width, marker_size = get_models_style()
    for k in metrics_style.keys():
        metrics_display[metrics_style[k]['name']] = metrics_style[k]['legend']
        metrics.append(metrics_style[k]['name'])
    metrics_display['y_true'] = 'GT'

    # Read URLs
    id_url = utils.image_to_url(files['images_path'])
    images = list(set(image_metrics.ImageID.values.tolist()))

    f = open(out_filename, 'w')
    f.write('<html><body><table border ="2">')
    for c, im in enumerate(images):
        url = id_url[im]
        f.write('<tr><td>' + str(c) + '</td>')
        f.write('<td><img src ="' + str(url) + '" width=200></td><td>')
        im_df = image_metrics.loc[(image_metrics.ImageID == im), :]
        f.write(im_df.to_html(border=4, index=False, justify='center',
                              col_space=60).encode('utf-8', errors='ignore'))
        f.write('</td><td>')
        for metric in metrics:
            metric_name = metrics_display[metric]
            # Return the label that was ranked first by the metric.
            l_first = im_df.sort_values(by=metric, ascending=False)[
                'DisplayName'].iat[0]
            l_true = im_df.sort_values(by='y_true', ascending=False)[
                'DisplayName'].iat[0]
            f.write('<b>' + metric_name + '</b> : ' + l_first)
            # Print a red cross if conf. H is correct.
            if metric == 'cH' and l_first == l_true:
                f.write('<font color="red"><b> + </b></font>')
            f.write('<br>')
        f.write('</td>')
        f.write('</td></tr>')
    f.write('</table></body></html>')
    f.close()
    print('Finished writing html to [%s]' % utils.blue(out_filename))


def oid_px_vs_entropy(singles, num_images):
    font_size = 'x-large'
    p, h, xy = OrderedDict(), OrderedDict(), OrderedDict()
    for k, v in singles.items():
        px = float(v)/float(num_images)
        p[k] = px
        h[k] = -px*np.log2(px)-(1-px)*np.log2(1-px)
        xy[k] = (p[k], h[k])
    x_val = [x[0] for x in xy.values()]
    y_val = [x[1] for x in xy.values()]
    plt.scatter(x_val, y_val)
    plt.xlabel('p', fontsize=font_size)
    plt.ylabel('H(p)', fontsize=font_size)
    plt.show()


# I/O - iota GT file path -> levels of agreement between raters.
def plot_raters_agreement(iota_fn):
    gt, gt_majority = dict(), dict()
    iota = pd.read_csv(iota_fn, usecols=range(1, 4))
    images = list(set(iota.ImageID.values.tolist()))
    print('IOTA-10K evaluation set has [%d] images' % len(images))
    for im in images:
        gt[im] = iota.loc[iota.ImageID == im, 'L1'].tolist()

    for i, l in gt.items():
        c = Counter(l)
        labels = [k for k, v in c.items() if v >= 2 and '/m/'
                  in k]
        if not labels: continue
        # In case of tie - choose at random.
        gt_label = np.random.choice(labels) if len(labels) > 1 else ''.join(
            labels)
        gt_majority[i] = gt_label


def plot_robustness_to_num_trees(hp):
    trees = [1,3,5,10]
    print('Computing for trees: %s' % str(trees))

    fig, ax = plt.subplots(figsize=(8, 5))
    tree_to_p = dict()
    for num_trees in trees:
        hp['seed'] = num_trees
        param_string = utils.create_param_string(hp, True)
        path = 'Results/' + param_string + '/results.pkl'
        print('Read results from %s' % path)

        (precision, sem_p, recall, sem_r) = pickle.load(open(path, 'r'))
        tree_to_p[num_trees] = precision['cH'][0][0]

    plt.plot(tree_to_p.keys(), tree_to_p.values(), 'sr', markersize=7)
    ax.set_xlabel('number of trees', fontsize=18)
    ax.set_ylabel('$cw\Delta{H}$ precision', fontsize=18)
    ax.set_xlim(left=0, right=num_trees+1)
    fn = 'Results/Robust/tree' + param_string +'.png'
    plt.savefig(fn, bbox_inches='tight', facecolor='white')
    print('Write results to %s' % fn)


def plot_robustness_to_vocab_size(hp, atleast, show=False):
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    atleast_to_voc, atleast_to_nImages, atleast_to_precision = \
        OrderedDict(), dict(), dict()
    metrics, fs, lw, ms = get_models_style()
    hp['seed'] = hp['max_seed']

    tf_to_cH, tf_to_cDKL, tf_to_cSingleton, tf_to_cPX, tf_to_cMI, \
    tf_to_confidence = dict(), dict(), dict(), dict(), dict(), dict()

    for t in atleast:
        hp['atleast'] = t

        param_string = utils.create_param_string(hp, True)

        # Read precision
        path_results = 'Results/robust/vocab/' + param_string + '/results.pkl'
        print('Read results from %s' % path_results)
        (precision, sem_p, recall, sem_r) = pickle.load(open(path_results, 'r'))
        atleast_to_precision[t] = {'cH':precision['cH'][0][0],
                                   'cDKL': precision['cDKL'][0][0],
                                   'cSingleton': precision['cSingleton'][0][0],
                                   'cPX': precision['cPX'][0][0],
                                   'cMI': precision['cMI'][0][0],
                                   'Confidence': precision['Confidence'][0][0]}

        tf_to_cH[t] = precision['cH'][0][0]
        tf_to_cDKL[t] = precision['cDKL'][0][0]
        tf_to_cSingleton[t] = precision['cSingleton'][0][0]
        tf_to_cPX[t] = precision['cPX'][0][0]
        tf_to_cMI[t] = precision['cMI'][0][0]
        tf_to_confidence[t] = precision['Confidence'][0][0]

        # Read vocabulary size.
        path_voc = 'Results/robust/vocab/' + param_string + \
                   '/avg_label_metrics.pkl'
        label_metrics = pickle.load(open(path_voc, 'r'))
        atleast_to_voc[t] = label_metrics.shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    voc_size = atleast_to_voc.values()
    plt.plot(voc_size, tf_to_cH.values(), color=colors['royalblue'],
             markersize=ms, marker=metrics[0]['marker'],
             label=metrics[0]['legend'])

    plt.plot(voc_size, tf_to_cDKL.values(), '-', markersize=ms,
             color=colors['lightcoral'], marker=metrics[1]['marker'],
             label=metrics[1]['legend'])

    plt.plot(voc_size, tf_to_cMI.values(), '-', markersize=ms,
             marker=metrics[2]['marker'], color=colors['skyblue'],
             label=metrics[2]['legend'])

    plt.plot(voc_size, tf_to_cSingleton.values(), '-', markersize=ms,
             marker=metrics[3]['marker'], color=colors['mediumpurple'],
             label=metrics[3]['legend'])

    plt.plot(voc_size, tf_to_cPX.values(), '-', markersize=ms,
             marker=metrics[4]['marker'], color=colors['palegreen'],
             label=metrics[4]['legend'])

    plt.plot(voc_size, tf_to_confidence.values(), '-', markersize=ms,
             marker=metrics[5]['marker'], color=colors['orange'],
             label=metrics[5]['legend'])

    plt.box(on=None)

    ax.legend(loc='upper right', frameon=False, fontsize=fs['legend'],
                    ncol=3, bbox_to_anchor=(0.5, 0.8, 0.5, 0.5))
    ax.set_ylabel('precision@1', fontsize=fs['axis'])
    ax.set_xlabel('vocabulary size', fontsize=fs['axis'])
    plt.rc('xtick', labelsize=fs['ticks'])
    plt.rc('ytick', labelsize=fs['ticks'])

    if show: plt.show()
    fn = 'Results/robust/vocab/vocab_' + param_string + '.png'
    plt.savefig(fn, bbox_inches='tight', facecolor='white')
    print('Write results to %s' % fn)


def plot_network(sorted_graph, classes_fn, ind_to_label, mis_dict, ind=True):
    disp = utils.load_display_names(classes_fn)
    G = Digraph('tree', filename='tree.gv')
    G.attr(size='6,6')
    G.node_attr.update(color='grey', style='filled')

    for k, v in sorted_graph.items():
        for i in v:
            if len(v) == 0: continue
            G.edge(str(k)+disp[ind_to_label[k]], str(i)+disp[ind_to_label[i]])
    G.view()

