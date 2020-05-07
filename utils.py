"""Utility functions for IOTA

"""
import model
import argparse
from collections import Counter
from datetime import datetime
import json
import numpy as np
import os
import pickle
import pandas as pd
from termcolor import colored

np.set_printoptions(precision=4)


# Handle flags.
def parse_flags():
    parser = argparse.ArgumentParser(description='Implement IOTA.')

    # Specify the distribution on which we learn the metrics.
    parser.add_argument('--annotations', default='oid',
                        help='allowed values: oid')
    parser.add_argument('--skip_probability', default=0.1,
                        help='probability to skip a label in the data')
    parser.add_argument('--atleast', default=100,
                        help='threshold on label occurrence')
    parser.add_argument('--rater', default='machine',
                        help='allowed values: human, machine')
    parser.add_argument('--split', default='validation',
                        help='allowed values: train, validation, test')
    parser.add_argument('--max_seed', default=10,
                        help='enum over seeds 0,...,max_seed')
    parser.add_argument('--eval_set', default='iota_validation',
                        help='allowed values: iota, iota_validation, iota_test')
    parser.add_argument('--min_rater_count', default=2,
                        help='only consider ratings with that many raters agree')
    parser.add_argument('--plot_figures', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='if true, plot and save figures')
    parser.add_argument('--eval_method', default='sl',
                        help='allowed values: sl (single-label), '
                             'ml(multi-label)')
    parser.add_argument('--k', default=10,
                        help='precision@k value')

    # The following flags are implemented in compute_image_metrics().
    parser.add_argument('--do_verify', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Use verified labels if true')
    parser.add_argument('--y_force', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Consider only cases in which GT label'
                             ' is in the returned labels')
    parser.add_argument('--gt_vocab', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='When false, also includes images whose GT '
                             'not in vocabulary')
    args = parser.parse_args()

    assert (args.split in ['train', 'validation', 'test'])
    assert (args.rater in ['machine', 'human'])
    assert (args.min_rater_count > 0)
    assert (int(args.min_rater_count) < 4)
    assert (args.eval_set in ['iota_validation', 'iota_test'])
    assert (args.eval_method in ['sl', 'ml'])
    assert (float(args.skip_probability) < 1.0)
    assert (float(args.skip_probability) >= 0.0)
    assert (args.gt_vocab == args.do_verify)
    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_param_string(hp, add_eval_set=False):
    # Global parameters.
    param_string = '_'.join([hp['annotations'], hp['split'], hp['rater'],
                             str(hp['atleast'])])
    param_string += '_s' + str(hp['seed'])
    param_string += 'p' + str(hp['skip_probability'])

    # Add eval-set dependent parameters.
    if add_eval_set:
        param_string += '_' + hp['eval_set']
        param_string += '_ver' if hp['do_verify'] else '_unver'
        param_string += '_voc' if hp['gt_vocab'] else '_novoc'
        param_string += '_yforce' if hp['y_force'] else '_noyforce'
        param_string += '_' + hp['eval_method']
    return param_string


def set_file_path(hp):
    files = dict()
    param_string = create_param_string(hp, False)
    #
    # data
    #   |_ ground_truth
    #   |_oid
    #       |_classes
    #       |_train
    #       |_validation
    #       |_test
    # results
    #   |_models
    #   |_counts
    #
    results_dir = os.environ.get('RES_DIR') if 'RES_DIR' in os.environ else ''
    data_dir = os.environ.get('OID_DIR') if 'OID_DIR' in os.environ else 'data/'

    files['oid_dir'] = data_dir
    files['results_dir'] = results_dir
    files['counts_dir'] = files['results_dir'] + 'counts/'
    files['models_dir'] = files['results_dir'] + 'models/'

    files['annotations']=files['oid_dir']+hp['split']+'/annotations-'+hp['rater']+'.csv'
    files['eval_fn']=files['oid_dir']+hp['split']+'/annotations-'+hp['rater']+'.csv'
    files['ver'] = files['oid_dir'] + hp['split'] + '/annotations-human.csv'
    files['images_path'] = files['oid_dir'] + hp['split'] + '/images.csv'
    files['class_descriptions'] = files['oid_dir'] + 'classes/class-descriptions.csv'

    # Path to IOTA-10K.
    files['iota10k'] = files['oid_dir'] + 'evaluation/' + hp['eval_set'] +'.csv'
    files['gt_dir'] = 'data/ground_truth/'  # pkl

    # Output files.
    files['model_fn'] = '%smodel_%s.pkl' % (files['models_dir'], param_string)
    files['counts_fn'] = '%scount_%s.pkl' % (files['counts_dir'], param_string)
    files['metrics_fn'] = '%slabel_metrics_%s.pkl' % (files['models_dir'],
                                                      param_string)
    files['gt_fn'] = '%sgt_dict_%s_%s.pkl' % (files['gt_dir'],
                                              hp['eval_set'],
                                              hp['eval_method'])

    # Output file names, (depend on eval set)
    param_string_eval_set = create_param_string(hp, True)
    files['image_metrics_fn'] = ('%s%s/image_metrics_%s.pkl' %
                                 (files['results_dir'],
                                  param_string_eval_set,
                                  param_string_eval_set))
    return files


def blue(s): return colored(s, 'blue')


def red(s): return colored(s, 'red')


def green(s): return colored(s, 'green')


def timestamp(index, total=-1):
    stamp = datetime.now().strftime('%H:%M:%S.%f')[:-4]
    if total < 0:
        return stamp
    else:
        return ('index=%d/%d %s' % (index, total, stamp))


def print_top_labels(k, dkls, ind_to_label, display_names):
    inds = np.argsort(dkls)
    for i in range(k):
        ind = inds[i]
        print(str(i) + ' dkl = ' + str(dkls[ind]) +
              '  ' + ind_to_label[ind] +
              '  ' + display_names[ind_to_label[ind]])


def parse_json(filepath):
    root = json.loads(open(filepath).read())
    return walk(root)


# Turn the JSON file to a dictionary {lbl, parent}
def walk(node, res={}):
    if 'children' in dict.keys(node):
        kids_list = node['children']
        for curr in kids_list:
            res.update({curr['name']: node['name']})
            walk(curr)
    else:
        return
    return res


# Map a label mid to its display name
def load_display_names(classes_filename):
    classes = pd.read_csv(classes_filename, names=['mid', 'name'])
    display_names = dict(zip(classes.mid, classes.name))
    return display_names


# Map { image id --> url }
def image_to_url(images_path):
    urls = pd.read_csv(images_path)
    id_url = dict(zip(urls.ImageID, urls.Thumbnail300KURL))
    return id_url


# Parse a DF into a dict {image -> associated labels}
def image_to_labels(annotations):
    img_to_labels, col_name = dict(), 'ImageID'
    images = annotations[col_name].unique().tolist()
    for i in images:
        img_to_labels[i] = annotations[annotations[col_name] == i][
            'LabelName'].values.tolist()
    return image_to_labels


def compute_metrics_baseline(metrics, singles, num_images, classes_fn):
    display_names = load_display_names(classes_fn)
    metrics['DisplayName'] = metrics['LabelName'].apply(lambda x: display_names[x] if x in display_names.keys() else '')
    metrics['px'] = metrics['LabelName'].apply(
        lambda x: float(singles[x]) / num_images)
    metrics['singleton'] = metrics['px'].apply(
        lambda x: model.entropy(np.array([1.0 - x, x])))
    return metrics


def compute_metrics_entropy_and_dkl(metrics, num_labels, singles ,
                                  mis_tree_graph,
                     num_images, c22_dict, ind_to_label):
    # =============================================
    # Compute delta entropy and DKLs given labels.
    # =============================================
    for start in range(num_labels):  # Run over labels in vocabulary.
        sorted_tree = model.sort_graph(mis_tree_graph, start)
        root_p = float(singles[ind_to_label[start]]) / num_images
        root_marginals = np.array([1.0 - root_p, root_p])
        root_dkl = model.compute_dkl(np.array([0.0, 1.0]), root_marginals)
        root_marginals = np.array([0.0, 1.0])
        H, _, Hm, dkl = model.update_marginals(sorted_tree, c22_dict,
                                               start, ind_to_label,
                                               root_marginals)
        if start % 100 == 0:
            print('label=%3d/%d H=%4.2f dkl=%4.2f root_dkl=%4.2f' %
                  (start, num_labels, H, dkl, root_dkl))
        metrics.loc[start, 'H'] = H
        metrics.loc[start, 'dkl'] = dkl + root_dkl
    return metrics


def compute_metrics_image_mi(metrics, singles, num_images):
    # =======================================================================
    # Compute information about image distribution MI(T; Image) is simply
    # computed from the fraction of images that contain a label.
    # =======================================================================
    singles_mi = model.compute_image_mi(singles, num_images)
    df = pd.DataFrame.from_dict(singles_mi, 'index', columns=['mi'])
    df.index.name = 'LabelName'
    metrics_df = pd.merge(metrics, df, how='outer', left_on='LabelName',
                      right_index=True)
    return metrics_df


def intialize_df(metrics_fn, ind_to_label):
    print('Compute metrics (missing [%s])' % red(metrics_fn))
    metrics = pd.DataFrame.from_dict(ind_to_label, 'index',
                                     columns=['LabelName'])
    metrics.index.name = 'index'
    return metrics


def save_metrics(metrics, metrics_fn):
    path = 'Data/models'
    if not os.path.exists(path): os.makedirs(path)
    pickle.dump(metrics, open(metrics_fn, 'wb'))
    print('Saved metrics df to [%s]' % green(metrics_fn))


def compute_metrics(ind_to_label, singles, c22_dict, mis_tree_graph,
                    num_images, num_labels, classes_fn, metrics_fn):
    # Load if exist.
    if os.path.isfile(metrics_fn):
        print('Load metrics df from [%s]' % blue(metrics_fn))
        metrics_df = pickle.load(open(metrics_fn, 'r'))
        return metrics_df

    # compute metrics
    metrics = intialize_df(metrics_fn, ind_to_label)
    metrics = compute_metrics_baseline(metrics, singles, num_images, classes_fn)
    metrics = compute_metrics_entropy_and_dkl(metrics, num_labels, singles,
                                      mis_tree_graph, num_images, c22_dict,
                                      ind_to_label)
    metrics = compute_metrics_image_mi(metrics, singles, num_images)

    add_metric_random(metrics)
    save_metrics(metrics, metrics_fn)
    return metrics


# Compute metrics (over the mixture model)
def compute_metrics_for_tree_mix(models, hp, files):
    param_string = create_param_string(hp, True)
    avg_label_metrics_fn = '%s%s/avg_label_metrics.pkl' % \
                           (files['results_dir'], param_string)
    if os.path.isfile(avg_label_metrics_fn):
        metrics_df = pickle.load(open(avg_label_metrics_fn, 'r'))
    else:
        metrics_per_seed = dict()
        for seed in range(len(models)):
            metrics_per_seed[seed] = models[seed]['label_metrics']
        metrics_df = average_metrics(metrics_per_seed)
        pickle.dump(metrics_df, open(avg_label_metrics_fn, 'w'))
    return metrics_df


# Average the scores over multiple seeds.
def average_metrics(metrics_per_seed):
    metrics_df = []
    for seed in range(len(metrics_per_seed)):
        metrics = metrics_per_seed[seed]
        metrics.set_index('LabelName', inplace=True)
        metrics.drop(columns='DisplayName', inplace=True)
        metrics_df.append(metrics)
    return sum(metrics_df) / (seed + 1)


# Add a random metric (if already exists, re-assign its values)
def add_metric_random(metrics_df):
    num_rows = metrics_df.shape[0]
    metrics_df['random'] = np.random.random(num_rows)
    return metrics_df


# Compute image-dependent metrics, based on label metrics and confidence.
# I/O: Construct a data-frame of the evaluation-set images with the models.
# Headers: image | label | px | singleton | H | dkl | mi | random | wH | y_true
def compute_image_metrics(image_to_gt, label_metrics, files, method='sl',
                          gt_in_voc=True, do_verify=False, y_force=True):
    # Load image-metrics.
    output_fn, eval_fn = files['image_metrics_fn'], files['eval_fn']
    if os.path.isfile(output_fn):
        print('Load image metrics [%s].' % blue(output_fn))
        return pickle.load(open(output_fn, 'r'))
    print('Missing [%s], compute.' % red(output_fn))

    # Read annotations to DF and match labels with its metrics.
    raw_ann = pd.read_csv(eval_fn)
    gtdf = pd.merge(raw_ann[['ImageID', 'LabelName', 'Confidence']],
                    label_metrics, how='left', on='LabelName')

    # Leave only images with GT.
    gtdf = gtdf[gtdf['ImageID'].isin(image_to_gt.keys())]

    # Leave only verified predictions.
    if do_verify and 'oid' in eval_fn:
        v = pd.read_csv(files['ver']).drop(columns=[
            'Source']).rename(columns={'Confidence': 'Verification'})
        gtdf = pd.merge(gtdf, v, on=['ImageID', 'LabelName'])
        gtdf = gtdf[gtdf['Verification'].isin([1])]

    # Compute image dependent metrics.
    gtdf.loc[:, 'cH'] = gtdf.Confidence * gtdf.H
    gtdf.loc[:, 'cDKL'] = gtdf.Confidence * gtdf.dkl
    gtdf.loc[:, 'cMI'] = gtdf.Confidence * gtdf.mi
    gtdf.loc[:, 'cPX'] = gtdf.Confidence * gtdf.px
    gtdf.loc[:, 'cSingleton'] = gtdf.Confidence * gtdf.singleton

    # Leave only labels in vocabulary.
    if gt_in_voc: gtdf = gtdf.dropna(how='any').copy(deep=True)

    disp = load_display_names(files['class_descriptions'])
    for k, v in disp.items(): disp[k] = v.lower()
    gtdf['DisplayName'] = gtdf.LabelName.apply(lambda x: disp[x] if x in
                                                                    disp.keys() else 'none')
    gtdf.loc[:, 'y_true'] = [0.0] * gtdf.shape[0]

    # Multiple labels.
    if method == 'ml':
        gtdf.loc[:, 'R1'] = gtdf.ImageID.apply(lambda x: image_to_gt[x][0])
        gtdf.loc[:, 'R2'] = gtdf.ImageID.apply(lambda x: image_to_gt[x][1])
        gtdf.loc[:, 'R3'] = gtdf.ImageID.apply(lambda x: image_to_gt[x][2])
        gtdf.loc[:, 'R1'] = (gtdf.LabelName == gtdf.R1).astype('int')
        gtdf.loc[:, 'R2'] = (gtdf.LabelName == gtdf.R2).astype('int')
        gtdf.loc[:, 'R3'] = (gtdf.LabelName == gtdf.R3).astype('int')
        gtdf.loc[:, 'y_true'] = gtdf.loc[:, ['R1', 'R2', 'R3']].mean(axis=1)
        gtdf.loc[:, 'y_true'] = gtdf.y_true.apply(lambda x: (x > 0).real)
    # Single Label.
    else:
        gtdf.loc[:, 'y_true'] = gtdf.ImageID.apply(lambda x: image_to_gt[x])
        gtdf.loc[:, 'y_true'] = (gtdf.LabelName == gtdf.y_true).astype(
            'int')

    # Drop images that y_true is not in its returned labels.
    is_gt_returned = gtdf.groupby(['ImageID'])['y_true'].any().to_dict()
    y_false = [k for k in is_gt_returned.keys() if is_gt_returned[k] is False]
    if y_force: gtdf = gtdf[~gtdf['ImageID'].isin(y_false)]

    # Drop rows with labels that are not in vocabulary.
    gtdf = gtdf.dropna(how='any').copy(deep=True)

    # Shuffle to avoid labels with the same confidence sorted the same way.
    indperm = np.random.permutation(gtdf.index)
    gtdf = gtdf.reindex(indperm)
    gtdf = gtdf.sort_values(by='ImageID', ascending=False).reset_index(drop=True)

    # Save results.
    dir = '/'.join(output_fn.split('/')[:-1])
    if not os.path.exists(dir): os.makedirs(dir)
    pickle.dump(gtdf, open(output_fn, 'w'))
    print('Saved metrics to [%s]' % green(output_fn))
    return gtdf


def precision_at_k(k, y_sorted):
    y_true = [0.0] * k
    idx = np.where(y_sorted > 0)[0].tolist()
    for i in idx:
        if i < k: y_true[i] = 1
    y_true = np.array(y_true)
    for c, i in enumerate(idx):
        y_true[i:] = c + 1
    return np.divide(y_true.tolist(), np.arange(1, k + 1, dtype=float))


def recall_at_k(k, y_sorted):
    y_true = [0.0] * k
    idx = np.where(y_sorted > 0)[0].tolist()
    for i in idx:
        if i < k: y_true[i] = 1
    y_true = np.array(y_true)
    for c, i in enumerate(idx):
        y_true[i:] = c + 1
    num_answer = 1 if sum(y_sorted) == 0 else sum(y_sorted)
    return np.divide(y_true.tolist(), float(num_answer))


def compute_recall(df, k):
    print('compute_recall...')
    average_recall, recall_mat, sem_rk = dict(), dict(), dict()
    jpgs = df.ImageID.unique()
    headers = df.columns.tolist()
    remove = ['ImageID', 'LabelName', 'DisplayName', 'y_true',
              'Verification', 'label_count', 'R1', 'R2', 'R3']
    metrics = [val for val in headers if val not in remove]

    n = len(jpgs)
    for metric in metrics:
        average_recall[metric] = np.zeros((1, k))
        recall_mat[metric] = np.zeros(shape=(n, k))

    # r@k is averaged over images.
    i = 0
    for index, image in enumerate(jpgs):
        if index % 1000 == 0: print(timestamp(index, len(jpgs)))
        image_df = df.loc[(df.ImageID == image), :]
        for metric in metrics:
            image_sorted_df = image_df[['DisplayName', metric, 'y_true']]
            image_sorted_df = image_sorted_df.sort_values(by=metric,
                                                          ascending=False)
            y = image_sorted_df['y_true'].values
            r = recall_at_k(k, y)
            recall_mat[metric][i] = r
            average_recall[metric] = np.add(average_recall[metric], np.array(r))
        i = i + 1
    for metric in metrics:
        average_recall[metric] = np.divide(average_recall[metric], n)
        sem_rk[metric] = np.divide(np.std(recall_mat[metric], axis=0),
                                   np.sqrt(n))
    return average_recall, sem_rk, recall_mat


def compute_precision(df, k):
    print('compute_precision...')
    average_precision, precision_mat, sem_pk = dict(), dict(), dict()

    jpgs = df.ImageID.unique()
    headers = df.columns.tolist()
    remove = ['ImageID', 'LabelName', 'DisplayName', 'y_true',
              'Verification', 'label_count', 'R1', 'R2', 'R3']
    metrics = [val for val in headers if val not in remove]

    num_images = len(jpgs)
    for metric in metrics:
        average_precision[metric] = np.zeros((1, k))
        precision_mat[metric] = np.zeros(shape=(num_images, k))

    # P@k is averaged over images.
    for i, image in enumerate(jpgs):
        if i % 1000 == 0: print(timestamp(i, len(jpgs)))
        image_df = df.loc[(df.ImageID == image), :]
        # This code again does many heavy searches ==> optimize.
        for metric in metrics:
            image_sorted_df = image_df[['DisplayName', metric, 'y_true']]
            image_sorted_df = image_sorted_df.sort_values(by=metric,
                                                          ascending=False)
            y = image_sorted_df['y_true'].values
            p = precision_at_k(k, y)
            precision_mat[metric][i] = p
            average_precision[metric] = np.add(average_precision[metric],
                                               np.array(p))
    for metric in metrics:
        average_precision[metric] = np.divide(average_precision[metric],
                                              num_images)
        sem_pk[metric] = np.divide(np.std(precision_mat[metric], axis=0),
                                   np.sqrt(num_images))
    return average_precision, sem_pk, precision_mat


def parse_iota10K(iota, min_rater_count=2):
    gt = dict()
    images = list(set(iota.ImageID.values.tolist()))
    print('IOTA-10K evaluation set has [%d] images' % len(images))
    for im in images:
        gt[im] = iota.loc[iota.ImageID == im, 'LabelName'].tolist()
    gt_majority = majority_vote(gt, min_rater_count)
    return gt_majority, gt


# Choose a single GT label by majority.
# I/O: {image -> [L1, L2, L3]} / {image -> L}.
def majority_vote(gt, min_rater_count):
    gt_majority = dict()
    for i, l in gt.items():
        c = Counter(l)
        labels = [k for k, v in c.items() if v >= min_rater_count and '/m/'
                  in k]
        if not labels: continue
        # In case of tie - choose at random.
        gt_label = np.random.choice(labels) if len(labels) > 1 else ''.join(
            labels)
        gt_majority[i] = gt_label
    return gt_majority


# Load dictionary mapping { id -> ground truth (BaC) label }
def load_evaluation_set(hp, eval_path, gt_dict_path, min_rater_count):
    if os.path.isfile(gt_dict_path):
        print('Load ground-truth dictionary from [%s]' % blue(gt_dict_path))
        image_to_gt = pickle.load(open(gt_dict_path, 'r'))
        return image_to_gt
    df = pd.read_csv(eval_path)
    image_to_sl, image_to_ml = parse_iota10K(df, min_rater_count)
    image_to_gt = image_to_sl if (hp['eval_method'] == 'sl') \
        else image_to_ml
    pickle.dump(image_to_gt, open(gt_dict_path, 'w'))
    print('Saved evaluation-set dictionary to [%s]' % green(gt_dict_path))
    return image_to_gt


# Leave one out to compute raters precision.
# I/O: { image -> [l1,l2,l3] } / image metrics.
def raters_performance(images, gt_path):
    gt_path = gt_path.replace('_sl', '_ml')
    image_to_gt_labels = pickle.load(open(gt_path, 'r'))
    # Ground truth subset.
    for k, v in image_to_gt_labels.items():
        if k not in images: del image_to_gt_labels[k]
    print('Compute raters agreement.')
    image_to_p = dict()
    for k, v in image_to_gt_labels.items():
        agreement = Counter(v).values()  # image -> [3] / [2,1] / [1,2]
        if len(agreement) == 1: image_to_p[k] = 1
        if len(agreement) == 2: image_to_p[k] = 1.0 / 3.0
    print(len(image_to_p))
    p = sum(image_to_p.values()) / len(image_to_p)
    return p


def save_results(hp, results_dir, precision, sem_p, recall, sem_r):
    add_eval_set_to_string = True
    param_string = create_param_string(hp, add_eval_set_to_string)

    path = results_dir + param_string
    if not os.path.exists(path):
        os.makedirs(path)

    output_filename = '/precision_recall.pkl'
    pickle.dump((precision, sem_p, recall, sem_r), open(path +
                                                        output_filename, 'w'))


# Load train, test, validation image - url files into df.
def load_urls_to_df(path_train, path_val, path_test):
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)
    urls = pd.concat([df_train, df_val, df_test])
    urls.set_index('ImageID', inplace=True)
    return urls
