import pandas as pd
from engine import PredefinedFeatures
from unimol_tools import MolPredict
import joblib
from visualization import plot_transporter_chemical_space, plot_transporter_performance_freq
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

class TransporterEval:
    def __init__(self):
        self.transport_file = './data/eval_4_transporter/dataset/transporter_dataset.csv'
        self.n_neighbors = [20, 35, 50, 65, 80]
        self.min_dist = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
        self.n_repetition = 3
        self.num_folds = 5
    
    def get_features(self, ref_rdkit_train_file, save_file):
        transporter_descs = PredefinedFeatures.rdkit_descs_transporter(ref_rdkit_train_file, self.transport_file, save_file)
        return transporter_descs
    
    def lgbm_combined_eval(self):
        ref_rdkit_train_file='./data/eval_1_combined/predefined_descs/rdkit_train.csv'
        save_file='./data/eval_4_transporter/predefined_descs/combined_transporter_rdkit.csv'
        transporter = pd.read_csv(self.transport_file)
        transporter_descs = self.get_features(ref_rdkit_train_file, save_file)
        features = transporter_descs.drop(columns=['SMILES', 'logPapp'])
        true_values = transporter_descs['logPapp']
        results = pd.DataFrame()
        error = pd.DataFrame()
        for rep in range(self.n_repetition):
            fold_pred = pd.DataFrame()
            for fold in range(self.num_folds):
                model = joblib.load(f'./trained_models/eval_1_combined/lightgbm/rep_{rep}/rdkit_trained_model_fold_{fold}.pkl')
                y_pred = model.predict(features)
                fold_pred[f'fold_{fold}'] = y_pred
            avg_pred = fold_pred.mean(axis=1)
            results[f'rep_{rep}'] = avg_pred
            error[f'rep_{rep}'] = abs(true_values - avg_pred)
        results['logPapp'], results['SMILES'], results['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
        error['logPapp'], error['SMILES'], error['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
        results['mean_pred'], results['std'] = results.iloc[:,:3].mean(axis=1), results.iloc[:,:3].std(axis=1)
        error['mean_error'], error['std'] = error.iloc[:,:3].mean(axis=1), error.iloc[:,:3].std(axis=1)
        results.to_csv('./prediction_analysis/eval_4_results/combined/lgbm_predictions.csv', index=False)
        error.to_csv('./prediction_analysis/eval_4_results/combined/lgbm_abs_error.csv', index=False)
    
    def lgbm_lit_eval(self):
        lit_names = ['wang_2016', 'wang_2020', 'wang_chen', 'pytdc']
        transporter = pd.read_csv(self.transport_file)
        true_values = transporter['logPapp']
        results = pd.DataFrame()
        error = pd.DataFrame()
        for name in lit_names:
            ref_rdkit_train_file=f'./data/eval_3_literature/predefined_descs/{name}_rdkit_train.csv'
            save_file=f'./data/eval_4_transporter/predefined_descs/{name}_transporter_rdkit.csv'
            transporter_descs = self.get_features(ref_rdkit_train_file, save_file)
            features = transporter_descs.drop(columns=['SMILES', 'logPapp'])
            true_values = transporter_descs['logPapp']
            model = joblib.load(f'./trained_models/eval_3_literature/lightgbm/{name}/rdkit_trained_model.pkl')
            y_pred = model.predict(features)
            if name == 'wang_2020':
                results[name] = y_pred
                error[name] = abs(true_values - y_pred)
            else:
                results[name] = y_pred + 6 # Add 6 as this is the simplified expression for log((10**y_pred)*(10**6))
                error[name] = abs(true_values - (y_pred + 6))
        results['logPapp'], results['SMILES'], results['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
        error['logPapp'], error['SMILES'], error['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
        results.to_csv('./prediction_analysis/eval_4_results/literature/lgbm_predictions.csv', index=False)
        error.to_csv('./prediction_analysis/eval_4_results/literature/lgbm_abs_error.csv', index=False)
    
    def unimol_combined_eval(self):
        transporter = pd.read_csv(self.transport_file)
        true_values = transporter['logPapp']
        results = pd.DataFrame(index=range(len(transporter)))
        error = pd.DataFrame(index=range(len(transporter)))
        for rep in range(self.n_repetition):
            model = MolPredict(load_model=f'./trained_models/eval_1_combined/unimol/rep_{rep}')
            res = model.predict(self.transport_file)
            results[f'rep_{rep}'] = res
            error[f'rep_{rep}'] = abs(true_values - results[f'rep_{rep}'])
        results['logPapp'], results['SMILES'], results['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
        error['logPapp'], error['SMILES'], error['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
        results['mean_pred'], results['std'] = results.iloc[:,:3].mean(axis=1), results.iloc[:,:3].std(axis=1)
        error['mean_error'], error['std'] = error.iloc[:,:3].mean(axis=1), error.iloc[:,:3].std(axis=1)
        results.to_csv('./prediction_analysis/eval_4_results/combined/unimol_predictions.csv', index=False)
        error.to_csv('./prediction_analysis/eval_4_results/combined/unimol_abs_error.csv', index=False)
    
    def unimol_literature_eval(self):
        lit_names = ['wang_2016', 'wang_2020', 'wang_chen', 'pytdc']
        transporter = pd.read_csv(self.transport_file)
        true_values = transporter['logPapp']
        results = pd.DataFrame(index=range(len(transporter)))
        error = pd.DataFrame(index=range(len(transporter)))
        for rep in range(self.n_repetition):
            for name in lit_names:
                model = MolPredict(load_model=f'./trained_models/eval_3_literature/unimol/rep_{rep}/{name}')
                res = model.predict(self.transport_file)
                if name == 'wang_2020':
                    results[name] = res
                    error[name] = abs(true_values - results[name])
                else:
                    results[name] = res
                    results[name] = results[name] + 6
                    error[name] = abs(true_values - results[name])
            results['logPapp'], results['SMILES'], results['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
            error['logPapp'], error['SMILES'], error['Transporter'] = transporter['logPapp'], transporter['SMILES'], transporter['Transporter']
            results.to_csv(f'./prediction_analysis/eval_4_results/literature/unimol_predictions_rep_{rep}.csv', index=False)
            error.to_csv(f'./prediction_analysis/eval_4_results/literature/unimol_abs_error_rep_{rep}.csv', index=False)

    def performance_w_different_datasets(self, plot_only=True):
        lit_cols = ['wang_2016', 'wang_2020', 'wang_chen', 'pytdc']
        lgbm_combined = pd.read_csv('./prediction_analysis/eval_4_results/combined/lgbm_abs_error.csv')
        lgbm_lit = pd.read_csv('./prediction_analysis/eval_4_results/literature/lgbm_abs_error.csv')
        lgbm_combined['lgb upper limit'] = lgbm_combined['mean_error'] + lgbm_combined['std']
        lgbm_lit = lgbm_lit[lit_cols]
        lgbm_error = pd.concat([lgbm_combined[['SMILES', 'lgb upper limit']], lgbm_lit], axis=1)
        unimol_combined = pd.read_csv('./prediction_analysis/eval_4_results/combined/unimol_abs_error.csv')
        unimol_combined['unimol upper limit'] = unimol_combined['mean_error'] + unimol_combined['std']
        unimol_lit = pd.DataFrame()
        for i in range(3):
            lit = pd.read_csv(f'./prediction_analysis/eval_4_results/literature/unimol_abs_error_rep_{i}.csv')
            unimol_lit = pd.concat([unimol_lit, lit[lit_cols]], axis=1)
            unimol_lit.rename(columns={'wang_2016': f'wang_2016_{i}', 'wang_2020': f'wang_2020_{i}', 'wang_chen': f'wang_chen_{i}', 'pytdc': f'pytdc_{i}'}, inplace=True)
        for lit in lit_cols:
            unimol_lit[f'{lit}_mean'] = unimol_lit[[f'{lit}_0', f'{lit}_1', f'{lit}_2']].mean(axis=1)
            unimol_lit[f'{lit}_std'] = unimol_lit[[f'{lit}_0', f'{lit}_1', f'{lit}_2']].std(axis=1)
            unimol_lit[lit] =  unimol_lit[f'{lit}_mean'] + unimol_lit[f'{lit}_std']
        unimol_error = pd.concat([unimol_combined[['SMILES', 'unimol upper limit']], unimol_lit[lit_cols]], axis=1)
        lgbm_list = []
        for n in lit_cols:
            both_performing = len(lgbm_error[(lgbm_error['lgb upper limit'] < 0.3) & (lgbm_error[n] < 0.3)])
            combined_performing = len(lgbm_error[(lgbm_error['lgb upper limit'] < 0.3) & (lgbm_error[n] > 0.3)])
            individual_performing = len(lgbm_error[(lgbm_error['lgb upper limit'] > 0.3) & (lgbm_error[n] < 0.3)])
            none_performing = len(lgbm_error[(lgbm_error['lgb upper limit'] > 0.3) & (lgbm_error[n] > 0.3)])
            lit_list = [none_performing, both_performing, combined_performing, individual_performing]
            lgbm_list.extend(lit_list)
        unimol_list = []
        for n in lit_cols:
            both_performing = len(unimol_error[(unimol_error['unimol upper limit'] < 0.3) & (unimol_error[n] < 0.3)])
            combined_performing = len(unimol_error[(unimol_error['unimol upper limit'] < 0.3) & (unimol_error[n] > 0.3)])
            individual_performing = len(unimol_error[(unimol_error['unimol upper limit'] > 0.3) & (unimol_error[n] < 0.3)])
            none_performing = len(unimol_error[(unimol_error['unimol upper limit'] > 0.3) & (unimol_error[n] > 0.3)])
            lit_list = [none_performing, both_performing, combined_performing, individual_performing]
            unimol_list.extend(lit_list)
        plot_transporter_performance_freq(lgbm_list, unimol_list)
        if plot_only == False:
            merged = pd.merge(lgbm_error, unimol_error, on='SMILES')
            return merged
    
    def select_poorest_mols(self):
        merged = self.performance_w_different_datasets(plot_only=False)
        poorest_mols = merged.loc[merged.iloc[:, 1:].gt(0.3).all(axis=1)]
        return poorest_mols['SMILES'].tolist()
    
    def select_best_mols(self):
        merged = self.performance_w_different_datasets(plot_only=False)
        best_mols = merged.loc[merged.iloc[:, [1,6]].lt(0.3).all(axis=1)]
        return best_mols['SMILES'].tolist()
    
    def nearest_molecules(self, transporter_descs, train_descs, selected_mols):
        data = pd.concat([train_descs, transporter_descs], axis=0)
        smi = data['SMILES'].tolist()
        data = data.drop(columns=['logPapp', 'SMILES'])
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        scaled_data['SMILES'] = smi
        closest_dict = {}
        for s in selected_mols:
            y = scaled_data[scaled_data['SMILES']==s]
            y = y.drop(columns=['SMILES'])
            y = np.array(y).reshape(1,-1)
            remaining_mols = [x for x in selected_mols if x != s]
            train_data = scaled_data[~scaled_data['SMILES'].isin(remaining_mols)]
            train_smi = train_data['SMILES']
            train_data = train_data.drop(columns=['SMILES'])
            nn = NearestNeighbors(n_neighbors=4, metric='euclidean')
            nn.fit(train_data)
            distances, indices = nn.kneighbors(y)
            closest_dict[s] = [train_smi[i] for i in indices[0, 1:4]]
        return closest_dict
    
    def chemical_space_analysis(self, show_all=False, poorest_mols=True):
        ref_rdkit_train_file='./data/eval_1_combined/predefined_descs/rdkit_train.csv'
        save_file='./data/eval_4_transporter/predefined_descs/combined_transporter_rdkit.csv'
        train_descs = pd.read_csv(ref_rdkit_train_file)
        transporter_descs = self.get_features(ref_rdkit_train_file, save_file)
        train_descs = pd.read_csv(ref_rdkit_train_file)
        cols_to_drop = transporter_descs.columns[transporter_descs.isna().any()].tolist()
        transporter_descs = transporter_descs.drop(columns=cols_to_drop)
        train_descs = train_descs.drop(columns=cols_to_drop)
        combined = pd.concat([train_descs, transporter_descs], axis=0)
        if poorest_mols==True:
            all_mols = self.select_poorest_mols()
            selected_mols = ['NS(=O)(=O)C1=C(Cl)C=C(NCC2=CC=CO2)C(=C1)C(O)=O', 
                             '[H][C@]12[C@H](OC(=O)C3=CC=CC=C3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)C4=CC=CC=C4)C4=CC=CC=C4)C(C)=C([C@@H](OC(C)=O)C(=O)[C@]1(C)[C@@H](O)C[C@H]1OC[C@@]21OC(C)=O)C3(C)C',
                             'CCC[C@@]1(CCC2=CC=CC=C2)CC(O)=C([C@H](CC)C2=CC=CC(NS(=O)(=O)C3=NC=C(C=C3)C(F)(F)F)=C2)C(=O)O1', 
                             'ClC1=CC=CC(N2CCN(CCCCOC3=CC=C4CCC(=O)NC4=C3)CC2)=C1Cl',
                             'CON=C(C(=O)N[C@H]1[C@H]2SCC(COC(N)=O)=C(N2C1=O)C(=O)OC(C)OC(C)=O)C1=CC=CO1', 
                             '[H][C@]12SCC(C[N+]3=CC=CC=C3)=C(N1C(=O)[C@H]2NC(=O)CC1=CC=CS1)C([O-])=O'
                             ]
        else:
            all_mols = self.select_best_mols()
            selected_mols = ['[H][C@]12CC[C@]3([H])[C@]([H])(C[C@@H](O)[C@]4(C)[C@H](CC[C@]34O)C3=CC(=O)OC3)[C@@]1(C)CC[C@@H](C2)O[C@H]1C[C@H](O)[C@H](O[C@H]2C[C@H](O)[C@H](O[C@H]3C[C@H](O)[C@H](O)[C@@H](C)O3)[C@@H](C)O2)[C@@H](C)O1', 
                             'CCN1C(=O)N(CC)C2=C(N(C)C(/C=C/C3=CC=C(OC)C(OC)=C3)=N2)C1=O',
                             'COC1=CC(=CC=C1OCCCN1CCC(CC1)C1=NOC2=CC(F)=CC=C12)C(C)=O', 
                             'FC(F)OC1=CC=C(C=C1OCC1CC1)C(=O)NC1=C(Cl)C=NC=C1Cl',
                             'Cc2c(CC(=O)O)c1cc(F)ccc1c2=Cc3ccc(S(C)=O)cc3', 
                             '[H][C@]12SCC(COC(C)=O)=C(N1C(=O)[C@H]2NC(=O)[C@H](N)C1=CC=CC=C1)C(O)=O'
                             ]
        net_mols = [x for x in transporter_descs['SMILES'].tolist() if x not in selected_mols]
        filtered_transporter_descs = transporter_descs[transporter_descs['SMILES'].isin(all_mols)]
        closest_dict = self.nearest_molecules(filtered_transporter_descs, train_descs, all_mols)
        label_map = {}
        if show_all == True:
            for key, value in closest_dict.items():
                label_map[key] = str(1)
        else:
            for s in selected_mols:
                label_map[s] = str(1)
        combined = pd.concat([train_descs, transporter_descs], axis=0)
        combined['group'] = combined['SMILES'].map(label_map).fillna('0')
        plot_transporter_chemical_space(data=combined, n_neighbors=20, min_dist=0.99, remove_mols=net_mols, show_all=show_all, seed=100)
        # lookup logPapp values for transporter and their nearest neighbors
        lookup = dict(zip(combined['SMILES'], combined['logPapp']))
        logpapp_dict = {lookup.get(k, None): [lookup.get(s, None) for s in v] for k, v in closest_dict.items()}
        return logpapp_dict
    
    def nearest_neighbors_best(self):
        ref_rdkit_train_file='./data/eval_1_combined/predefined_descs/rdkit_train.csv'
        save_file='./data/eval_4_transporter/predefined_descs/combined_transporter_rdkit.csv'
        train_descs = pd.read_csv(ref_rdkit_train_file)
        transporter_descs = self.get_features(ref_rdkit_train_file, save_file)
        cols_to_drop = transporter_descs.columns[transporter_descs.isna().any()].tolist()
        transporter_descs = transporter_descs.drop(columns=cols_to_drop)
        all_mols = self.select_best_mols()
        transporter_descs = transporter_descs[transporter_descs['SMILES'].isin(all_mols)]
        train_descs = train_descs.drop(columns=cols_to_drop)
        closest_dict = self.nearest_molecules(transporter_descs, train_descs, all_mols)
        combined = pd.concat([train_descs, transporter_descs], axis=0)
        lookup = dict(zip(combined['SMILES'], combined['logPapp']))
        logpapp_dict = {lookup.get(k, None): [lookup.get(s, None) for s in v] for k, v in closest_dict.items()}
        return logpapp_dict # Atleast one neighbor in each best performing cases do align with the true value, suggesting the neighbors in this chemical space can influence prediction
    
    def init_all(self):
        self.lgbm_combined_eval()
        self.lgbm_lit_eval()
        self.unimol_combined_eval()
        self.unimol_literature_eval()
        self.performance_w_different_datasets()
        self.chemical_space_analysis(poorest_mols=True)
        self.chemical_space_analysis(poorest_mols=False)

def main():
    evaluate = TransporterEval()
    evaluate.init_all()

if __name__ == "__main__":
    main()
