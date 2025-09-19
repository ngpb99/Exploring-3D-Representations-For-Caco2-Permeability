import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def plot_top_feature_performance(mae_dict, model_name, color):
    y = [i for i in mae_dict.values()]
    x = [i for i in range(len(y))]
    plt.plot(x, y, linewidth=1.0, color=color)
    plt.xlabel('Number of Top Features')
    plt.ylabel('MAE Score')
    plt.title(f'Performance of Varying Number of Top Features ({model_name})')

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
    plt.plot(x, y_1, color='red', linewidth=3.0)
    plt.plot(x, y_2, color='blue', linewidth=3.0)
    plt.scatter(x, y_1, color='red', linewidth=3.0)
    plt.scatter(x, y_2, color='blue', linewidth=3.0)
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
    plt.legend(['UniMol', 'LightGBM'], loc='upper right')
    plt.show()

def plot_transporter_performance_freq(lgbm_predictions_freq, unimol_predictions_freq):
    groups = ["Wang_2016", "Wang_2020", "Wang_Chen", "PyTDC"]
    categories = ["None", "Both", "Combined", "Literature"]
    bottom = np.array(lgbm_predictions_freq).reshape(4,4)
    top = np.array(unimol_predictions_freq).reshape(4,4)
    x = np.arange(len(groups)) 
    bar_width = 0.18     
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    hatches = ['/', '/', '/', '/']
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cat in enumerate(categories):
        # shift each category within a group
        positions = x + i * bar_width
        # Bottom bars
        ax.bar(positions, bottom[:, i], width=bar_width, 
               color=colors[i], edgecolor='black')
        # Top bars (stacked, same color but hatched)
        ax.bar(positions, top[:, i], width=bar_width, 
               bottom=bottom[:, i], color=colors[i], 
               hatch=hatches[i], edgecolor='black')
    # Formatting
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(groups)
    ax.set_title("Transporter-mediated Prediction Outcome Across Literature and Combined Dataset")
    ax.set_ylabel("Frequency")
    # Legends
    # LightGBM legend (bottom solid colors)
    lightgbm_patches = [mpatches.Patch(facecolor=colors[i], edgecolor='black',
                                       label=cat) for i, cat in enumerate(categories)]
    # UniMol legend (top hatched)
    unimol_patches = [mpatches.Patch(facecolor=colors[i], edgecolor='black',
                                     hatch=hatches[i], label=cat) for i, cat in enumerate(categories)]
    # Place legends
    legend1 = ax.legend(handles=lightgbm_patches, title="LightGBM",
                        loc="upper left", bbox_to_anchor=(0.85, 1))
    legend2 = ax.legend(handles=unimol_patches, title="Uni-Mol",
                        loc="upper left", bbox_to_anchor=(0.85, 0.7))
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.tight_layout()
    plt.show()

def plot_transporter_chemical_space(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', remove_mols=None, show_all=False, seed=None):
    color_map = {'0':'gray', '1':'#d62728'}
    data['color'] = data['group'].map(color_map)
    smiles = data['SMILES'].tolist()
    color = data['color']
    data = data.drop(columns=['logPapp', 'SMILES', 'group', 'color'])
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed
    )
    u = fit.fit_transform(scaled_data)
    if show_all == False:
        df_u = pd.DataFrame(u)
        df_u['SMILES'] = smiles
        df_u['color'] = color.tolist()
        df_u = df_u[~df_u['SMILES'].isin(remove_mols)]
        color = df_u['color']
        u = np.array(df_u.drop(columns=['SMILES', 'color']))
    plt.scatter(u[:,0], u[:,1], c=color)
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.ylim(-30, 35)
    plt.xlim(-30, 35)
    plt.title('Chemical Space of Transporter-Mediated Compounds within Training data', fontsize=12)
    legend_elements = [mpatches.Patch(color='#d62728', label='Transporter-mediated molecules'),
                       mpatches.Patch(color='gray', label='Training molecules')
                       ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()
