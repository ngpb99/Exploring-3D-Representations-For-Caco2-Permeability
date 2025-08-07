import matplotlib.pyplot as plt
import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import Draw
from IPython.display import display
from rdkit import Chem

class PlotEmbeddings:
    def process_data(self, embeddings_df):
        embeddings_df['group'] = pd.cut(embeddings_df['logPapp'], bins=[-float('inf'), 0, 1, float('inf')], labels=[1,2,3], right=False)
        group = embeddings_df['group']
        data = embeddings_df.drop(columns=['logPapp', 'group'])
        if 'SMILES' in data.columns:
            data.drop(columns=['SMILES'], inplace=True)
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        return group, scaled_data
    
    def draw_umap(self, embeddings_df, model_name, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', highlight_idx=None, seed=None):
        group, data = self.process_data(embeddings_df)
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=seed
        )
        u = fit.fit_transform(data)
        fig = plt.figure()
        unique_labels = np.unique(group)
        if n_components == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(u[:,0], u[:,1], c=group)
            if highlight_idx != None:
                ax.scatter(u[highlight_idx,0], u[highlight_idx,1], c='red')
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(u[:,0], u[:,1], u[:,2], c=group, s=100)
        else:
            raise ValueError('Specified components not included in this function')
        cbar = fig.colorbar(scatter)
        cbar.set_ticks(unique_labels)
        title = f'{model_name} (Neighbours: {n_neighbors}, Min_Dist: {min_dist})'
        plt.xlabel('UMAP_1')
        plt.ylabel('UMAP_2')
        plt.title(title, fontsize=12)
        plt.show()

def plot_predictions(data, title, x_label, y_label):
    c_groups = {1:'purple', 2:'green', 3:'gold'}
    colors = [c_groups[g] for g in data['group']]
    fig = plt.figure(figsize=(7,5), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    x = data[x_label]
    y = data[y_label]
    fit = [i for i in range(-3, 4)]
    ax.scatter(x, y, c=colors)
    ax.plot(fit, fit, color='maroon')
    ax.set_xlabel('logPapp')
    ax.set_ylabel('Predicted logPapp')
    ax.set_title(title)
    plt.show()

def plot_scaffolds(data, title, x_label, y_label, color):
    fig = plt.figure(figsize=(7,5), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    x = data[x_label]
    y = data[y_label]
    fit = [i for i in range(-3, 4)]
    ax.scatter(x, y, c=color)
    ax.plot(fit, fit, color='maroon')
    ax.set_xlabel('logPapp')
    ax.set_ylabel('Predicted logPapp')
    ax.set_title(title)
    plt.show()

def draw_mols(legends, mols, mols_per_row):
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends)
    display(img)

def plot_top_feature_performance(mae_dict, model_name):
    y = [i for i in mae_dict.values()]
    x = [i for i in range(len(y))]
    plt.plot(x, y, linewidth=1.0)
    plt.scatter(x, y, linewidth=0.5)
    plt.xlabel('Number of Top Features')
    plt.ylabel('MAE Score')
    plt.title(f'Performance of Varying Number of Top Features ({model_name})')
    plt.show()

def plot_frequency(train_freq, test_freq, split_num):
    x = [i for i in range(5)]
    y = list(train_freq.values())
    plt.bar(x, y)
    plt.xticks(x, labels=[i+1 for i in list(train_freq.keys())]) # All group labels are +1 for better visuals
    plt.ylim(0, 160)
    plt.title(f'Train Split {split_num}')
    plt.xlabel('Scaffold Group')
    plt.ylabel('Scaffold Frequency')
    plt.show()
    x2 = [i for i in range(5)]
    y2 = list(test_freq.values())
    plt.bar(x2, y2, color='red')
    plt.xticks(x2, labels=[i+1 for i in list(test_freq.keys())]) # All group labels are +1 for better visuals
    plt.ylim(0, 160)
    plt.title(f'Test Split {split_num}')
    plt.xlabel('Scaffold Group')
    plt.ylabel('Scaffold Frequency')
    plt.show()

def plot_diversity(train_group_freq, test_group_freq):
    bar_width = 1
    x = pd.Series([i for i in range(1,4)])
    plt.bar(x - (bar_width/3)/2, train_group_freq, align='center', width=bar_width/3, color='#1f77b4')
    plt.bar(x + (bar_width/3)/2, test_group_freq, align='center', width=bar_width/3, color='red')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.ylabel('No. of Groups')
    plt.title('Group Diversity Within Each Split')
    plt.xticks([i for i in range(1,4)], ['Split 1', 'Split 2', 'Split 3'])
    plt.show()

def plot_group_performance(delta_df, split_num, start_grp_idx, end_grp_idx):
    x = [i for i in range(10)]
    y_1 = delta_df['unimol_pred_delta'][start_grp_idx:end_grp_idx]
    y_2 = delta_df['lgbm_pred_delta'][start_grp_idx:end_grp_idx]
    y_3 = delta_df['rf_pred_delta'][start_grp_idx:end_grp_idx]
    plt.plot(x, y_1, color='red', linewidth=3.0)
    plt.plot(x, y_2, color='blue', linewidth=3.0)
    plt.plot(x, y_3, color='green', linewidth=3.0)
    plt.scatter(x, y_1, color='red', linewidth=3.0)
    plt.scatter(x, y_2, color='blue', linewidth=3.0)
    plt.scatter(x, y_3, color='green', linewidth=3.0)
    plt.ylim(0, 1.3)
    if split_num == 2:
        plt.ylim(0, 2.0)
    plt.xticks(x, delta_df.index[start_grp_idx:end_grp_idx] + 1)
    plt.xlabel('Group')
    plt.ylabel('Mean Absolute Error')
    if start_grp_idx == 0:
        plt.title(f"MAE for 10 Most Populated Groups in Split {split_num}")
    else:
        plt.title(f"MAE for 11th–20th Most Populated Groups in Split {split_num}")
    plt.legend(['UniMol', 'LightGBM', 'RF'], loc='upper right')
    plt.show()