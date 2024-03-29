#!/usr/bin/env python
import pickle
import argparse
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import numpy as np
import seaborn as sns

def build_dataframe(trials, bestid):
    data = {}
    data['iteration'] = [t['tid'] for t in trials]
    data['loss'] = [t['result']['loss'] for t in trials]

    for p, k in enumerate(trials[0]['misc']['vals'].keys()):
        data[k] = [t['misc']['vals'][k][0] for t in trials]

    df = pd.DataFrame(data)
    bestdf = df[df['iteration'] == bestid['tid']]
    return df, bestdf

def plot_scans(df, bestdf, trials, bestid, file):
    print('plotting scan results...')
    # plot loss
    nplots = len(trials[0]['misc']['vals'].keys())+1
    f, axs = plt.subplots(1, nplots, sharey=True, figsize=(50,10))

    axs[0].scatter(df.get('iteration'), df.get('loss'))
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].set_yscale('log')
    axs[0].scatter(bestdf.get('iteration'), bestdf.get('loss'))

    # plot features
    for p, k in enumerate(trials[0]['misc']['vals'].keys()):

        if k in ('learning_rate','alpha1','alpha2','beta1','beta2','SD_norm','lnzRef1','lnzRef2','reward_bkg_norm', 'frac_bkg', 'lambda_cycle', 'lambda_id_factor'):
            axs[p+1].scatter(df.get(k), df.get('loss'))
            if k in 'learning_rate':
                axs[p+1].set_xscale('log')
                axs[p+1].set_xlim([1e-5, 1])
        else:
            sns.violinplot(df.get(k), df.get('loss'), ax=axs[p+1], palette="Set2",cut=0.0)
            sns.stripplot(df.get(k), df.get('loss'), ax=axs[p+1], color='gray', alpha=0.4)
        axs[p+1].set_xlabel(k)
        axs[p+1].scatter(bestdf.get(k), bestdf.get('loss'), color='orange')

    plt.savefig(f'{file}', bbox_inches="tight")

def plot_correlations(df, file):
    print('plotting correlations...')
    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr(), mask=np.zeros_like(df.corr(), dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, vmax=1, vmin=-1, annot=True, fmt=".2f")
    plt.savefig(f'{file}', bbox_inches='tight')

def plot_pairs(df, file):
    print('plotting pairs')
    plt.figure(figsize=(50,50))
    sns.pairplot(df)
    plt.savefig(f'{file}', bbox_inches='tight')

#----------------------------------------------------------------------
def main(args):
    """Load trials and generate plots"""
    with open(args.trials, 'rb') as f:
        input_trials = pickle.load(f)

    print('Filtering bad scans...')
    trials = []
    best = 10000
    bestid = -1
    for t in input_trials:
        if t['state'] == 2:
            trials.append(t)
            if t['result']['loss'] < best:
                best = t['result']['loss']
                bestid = t
    print(f'Number of good trials {len(trials)}')
    pprint.pprint(bestid)

    # compute dataframe
    df, bestdf = build_dataframe(trials, bestid)

    # plot scans
    plot_scans(df, bestdf, trials, bestid, f'{args.trials}_scan.png')

    # plot correlation matrix
    plot_correlations(df, f'{args.trials}_corr.png')

    # plot pairs
    plot_pairs(df, f'{args.trials}_pairs.png')

#----------------------------------------------------------------------
if __name__ == "__main__":
    """read command line arguments"""
    parser = argparse.ArgumentParser(description='Train an ML groomer.')
    parser.add_argument('trials', help='Pickle file with trials.')
    args = parser.parse_args()
    main(args)
