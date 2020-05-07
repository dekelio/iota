""" Main script for IOTA, Informative object annotations

"""
import model
import utils
import vis
import os
import numpy as np
import pandas as pd
import pickle
from scipy.sparse.csgraph import minimum_spanning_tree
import sys
from termcolor import colored
np.set_printoptions(precision=4)
pd.set_option('precision', 2)

print(os.getcwd())
os.environ['RES_DIR'] = '/cortex/users/brachalior/storage/results/'
os.environ['OID_DIR'] = '/cortex/data/images/openimagesV3/oid/'

# Init: flags, version, hyper parameters
assert(sys.version_info[0] == 2)  # Python 2.7
assert(pd.__version__ == u'0.23.4')
hp = dict()
args = utils.parse_flags()
hp['annotations'] = args.annotations
hp['atleast'] = int(args.atleast)
hp['rater'] = args.rater
hp['split'] = args.split
hp['max_seed'] = int(args.max_seed)
hp['skip_probability'] = float(args.skip_probability)
hp['eval_set'] = args.eval_set
hp['eval_method'] = args.eval_method
hp['do_verify'] = args.do_verify
hp['plot_figures'] = args.plot_figures
hp['min_rater_count'] = args.min_rater_count
hp['y_force'] = args.y_force
hp['gt_vocab'] = args.gt_vocab  # True - drop the image if its gt not in vocab.
hp['k'] = int(args.k)

# Loop over seeds.
models = dict()
for seed in range(hp['max_seed']+1):
    hp['seed'] = seed

    # Set file path.
    files = utils.set_file_path(hp)
    param_string = utils.create_param_string(hp, False)

    # Load model.
    if os.path.isfile(files['model_fn']):
        print('Load model from: %s' % files['model_fn'])
        models[seed] = pickle.load(open(files['model_fn'], 'rb'))
        continue

    # Load labels and counts.
    (count, pair_count, mis_dict, c22_dict, num_images, label_to_ind,
     ind_to_label, singles, num_labels) = model.load_count_data(
         files['counts_fn'], files['annotations'], hp)

    # Build an MI-based MST.
    mis_csr = model.dict_to_sparse(mis_dict, label_to_ind)
    mis_tree_csr = minimum_spanning_tree(mis_csr)
    mis_tree_graph = model.csr_to_graph(mis_tree_csr)

    # Sort the graph so each node has a single father, compute entropy.
    start = 0
    sorted_tree = model.sort_graph(mis_tree_graph, start)
    entropy = model.compute_tree_entropy(sorted_tree, start, c22_dict, singles,
                                         num_images, ind_to_label)

    # Compute label - metrics.
    label_metrics = utils.compute_metrics(ind_to_label, singles,
                                          c22_dict, mis_tree_graph,
                                          num_images, num_labels,
                                          files['class_descriptions'],
                                          files['metrics_fn'])

    # Pack all model variables in a single variable.
    models[seed] = dict()
    models[seed]['singles'] = singles
    models[seed]['ind_to_label'] = ind_to_label
    models[seed]['label_to_ind'] = label_to_ind
    models[seed]['mis_tree_graph'] = mis_tree_graph
    models[seed]['c22_dict'] = c22_dict
    models[seed]['num_images'] = num_images
    models[seed]['num_labels'] = num_labels
    models[seed]['metrics_fn'] = files['metrics_fn']
    models[seed]['label_metrics'] = label_metrics
    pickle.dump(models[seed], open(files['model_fn'], 'wb'))

label_metrics = utils.compute_metrics_for_tree_mix(models, hp, files)

# Load ground truth data.
image_to_gt = utils.load_evaluation_set(hp, files['iota10k'],
                                        files['gt_fn'],
                                        args.min_rater_count)

# Arrange metrics for the gt labels in df.
image_metrics = utils.compute_image_metrics(image_to_gt, label_metrics, files,
                                            method=hp['eval_method'],
                                            do_verify=hp['do_verify'],
                                            gt_in_voc=hp['gt_vocab'],
                                            y_force=hp['y_force'])

images = list(set(image_metrics.ImageID.values.tolist()))
raters_ub = utils.raters_performance(images, files['gt_fn'])
print('Raters agreement: %s' % raters_ub)

# Compute precision & recall over all metrics.
precision, sem_p, precision_mat = utils.compute_precision(image_metrics, hp['k'])
recall, sem_r, recall_mat = utils.compute_recall(image_metrics, hp['k'])
vis.print_top_pr(precision, recall)
utils.save_results(hp, files['results_dir'], precision, sem_p, recall, sem_r)

# Plot precision, recall and correlation. Save specific examples to HTML.
if args.plot_figures:
    vis.plot_precision(hp, files, precision, sem_p, raters_ub)
    vis.plot_recall(hp, files, recall, sem_r)
    vis.plot_precision_vs_recall(hp, files, precision, recall, raters_ub)
    # vis.plot_correlation(hp, files, image_metrics)
    vis.write_models_to_html(image_metrics, hp, files)
