from engine import GraphData, LGBMTrainer, RFTrainer, CBERTaTrainer, GraphormerTrainer, UniMolTrainer
from visualization import plot_frequency, plot_diversity, plot_group_performance, PlotEmbeddings
from embeddings import Embeddings
import deepchem as dc
import pandas as pd
import numpy as np
import json
import shap
import joblib

class ScaffoldSplit:
    def __init__(self):
        self.data_file = './data/eval_1_combined/dataset/3D_optimizable_dataset.csv'
        self.test_cutoff = 1000
        self.seed = 0

    def save_data(self, data, train_idx, test_idx, train_file, test_file):
        flattened = [item for sublist in train_idx for item in sublist]
        train = data.iloc[flattened, :]
        test = data.iloc[test_idx, :]
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
    
    def scaffold_split(self):
        split_strategy = ['split_1', 'split_2', 'split_3']
        data = pd.read_csv(self.data_file)
        Xs = np.zeros(len(data))
        Ys = np.array(data['logPapp'])
        dataset = dc.data.DiskDataset.from_numpy(X=Xs, y=Ys, w=np.zeros(len(data)), ids=list(data['SMILES']))
        scaffoldsplitter = dc.splits.ScaffoldSplitter()
        scaffold = scaffoldsplitter.generate_scaffolds(dataset)
        first_idx = 0
        last_idx = len(scaffold) - 1
        for ss in split_strategy:
            test_idx = []
            scaffold_remaining = scaffold.copy()
            np.random.seed(self.seed)
            while len(test_idx) <= self.test_cutoff and ss == 'split_1':
                random = np.random.randint(0, len(scaffold_remaining))
                selected_group = scaffold_remaining.pop(random)
                test_idx.extend(selected_group)
            while len(test_idx) <= self.test_cutoff and ss == 'split_3':
                selected_group = scaffold_remaining.pop(first_idx)
                test_idx.extend(selected_group)
            while len(test_idx) <= self.test_cutoff and ss == 'split_2':
                selected_group = scaffold_remaining.pop(last_idx)
                test_idx.extend(selected_group)
                last_idx -= 1
            train_dir = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_dir = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            self.save_data(data, scaffold_remaining, test_idx, train_dir, test_dir)
        with open('./data/eval_2_scaffold/scaffold_idx_list.json', 'w') as f:
            json.dump(scaffold, f)
    




class ModelTrainer:
    def __init__(self):
        self.split_strategy = ['split_1', 'split_2', 'split_3']
        self.seed1 = 0
        self.seed2 = 123
        self.n_repetition = 3
    
    def get_features_rdkit(self, train_file, test_file):
        ref_train_rdkit = pd.read_csv('./data/eval_1_combined/predefined_descs/rdkit_train.csv')
        ref_test_rdkit = pd.read_csv('./data/eval_1_combined/predefined_descs/rdkit_test.csv')
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        reference_rdkit = pd.concat([ref_train_rdkit, ref_test_rdkit], axis=0)
        train_smiles, test_smiles = train['SMILES'], test['SMILES']
        train_rdkit = reference_rdkit[reference_rdkit['SMILES'].isin(train_smiles)].reset_index(drop=True)
        test_rdkit = reference_rdkit[reference_rdkit['SMILES'].isin(test_smiles)].reset_index(drop=True)
        return train_rdkit, test_rdkit
    
    def generate_graph(self, train_file, test_file, split_strategy):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        generator = GraphData()
        generator.save_graph(train, test, graph_dir=f'./data/2D_graph_data/eval_2_scaffold/{split_strategy}')
    
    def lgbm(self):
        params_dict = {}
        performance_dict = {}
        for ss in self.split_strategy:
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            train_rdkit, test_rdkit = self.get_features_rdkit(train_file, test_file)
            model_dir = f'./trained_models/eval_2_scaffold/lightgbm/{ss}'
            trainer = LGBMTrainer(train_rdkit, test_rdkit, seed=self.seed1)
            best_params = trainer.model_tuning()
            params_dict[ss] = best_params
            r2, mse, mae = trainer.model_testing('rdkit', best_params, model_dir)
            performance_dict[ss] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open('./trained_models/eval_2_scaffold/lightgbm/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
        with open('./trained_models/eval_2_scaffold/lightgbm/best_hyperparams.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    def lgbm_predictions(self):
        num_folds=5
        for ss in self.split_strategy:
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            train_rdkit, test_rdkit = self.get_features_rdkit(train_file, test_file)
            trainer = LGBMTrainer(train_rdkit, test_rdkit, seed=self.seed1)
            predictions_df = pd.DataFrame()
            for fold in range(num_folds):
                model_file = f'./trained_models/eval_2_scaffold/lightgbm/{ss}/rdkit_trained_model_fold_{fold}.pkl'
                pred = trainer.model_predict(model_file)
                predictions_df[f'fold_{fold}_pred'] = pred
            predictions_df['pred_logPapp'] = predictions_df.mean(axis=1)
            predictions_df['SMILES'] = test_rdkit['SMILES']
            predictions_df['logPapp'] = test_rdkit['logPapp']
            predictions_df.to_csv(f'./trained_models/eval_2_scaffold/lightgbm/{ss}/predictions.csv', index=False)
            
    def rf(self):
        params_dict = {}
        performance_dict = {}
        for ss in self.split_strategy:
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            train_rdkit, test_rdkit = self.get_features_rdkit(train_file, test_file)
            model_dir = f'./trained_models/eval_2_scaffold/rf/{ss}'
            trainer = RFTrainer(train_rdkit, test_rdkit, seed=self.seed1)
            best_params = trainer.model_tuning()
            params_dict[ss] = best_params
            r2, mse, mae = trainer.model_testing('rdkit', best_params, model_dir)
            performance_dict[ss] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open('./trained_models/eval_2_scaffold/rf/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
        with open('./trained_models/eval_2_scaffold/rf/best_hyperparams.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    def rf_predictions(self):
        num_folds=5
        for ss in self.split_strategy:
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            train_rdkit, test_rdkit = self.get_features_rdkit(train_file, test_file)
            trainer = RFTrainer(train_rdkit, test_rdkit, seed=self.seed1)
            predictions_df = pd.DataFrame()
            for fold in range(num_folds):
                model_file = f'./trained_models/eval_2_scaffold/rf/{ss}/rdkit_trained_model_fold_{fold}.pkl'
                pred = trainer.model_predict(model_file)
                predictions_df[f'fold_{fold}_pred'] = pred
            predictions_df['pred_logPapp'] = predictions_df.mean(axis=1)
            predictions_df['SMILES'] = test_rdkit['SMILES']
            predictions_df['logPapp'] = test_rdkit['logPapp']
            predictions_df.to_csv(f'./trained_models/eval_2_scaffold/rf/{ss}/predictions.csv', index=False)
    
    def chemberta(self):
        performance_dict = {}
        for rep in range(self.n_repetition):
            seed = self.seed2 + rep
            for ss in self.split_strategy:
                train = pd.read_csv(f'./data/eval_2_scaffold/{ss}_scaffold_train.csv')
                test = pd.read_csv(f'./data/eval_2_scaffold/{ss}_scaffold_test.csv')
                save_dir = f'./trained_models/eval_2_scaffold/chemberta-2/rep_{rep}/{ss}'
                trainer = CBERTaTrainer(train, test, model_name='DeepChem/ChemBERTa-77M-MTR')
                trainer.model_training(save_dir, train_batch_size=128, eval_batch_size=128, train_epoch=100, 
                                       eval_steps=10, max_length=195, seed=seed)
                r2, mse, mae = trainer.model_testing(save_dir, max_length=195)
                performance_dict[ss] = (
                    f'R2: {r2:.4f} '
                    f'MAE: {mae:.4f} '
                    f'MSE: {mse:.4f}'
                )
            with open(f'./trained_models/eval_2_scaffold/chemberta-2/rep_{rep}/model_performance.json', 'w') as f:
                json.dump(performance_dict, f, indent=2)
    
    def graphormer(self):
        performance_dict = {}
        for ss in self.split_strategy:
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            self.generate_graph(train_file, test_file, ss)
            trainer = GraphormerTrainer(data_dir=f'./data/2D_graph_data/eval_2_scaffold/{ss}', model_name='clefourrier/graphormer-base-pcqm4mv2')
            save_dir = f'./trained_models/eval_2_scaffold/graphormer/{ss}'
            trainer.model_training(save_dir, train_batch_size=128, eval_batch_size=128, train_epoch=100, 
                                   eval_steps=10, seed=self.seed2)
            r2, mse, mae = trainer.model_testing(save_dir, test_file)
            performance_dict[ss] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open('./trained_models/eval_2_scaffold/graphormer/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)

    # Seed for Uni-Mol needs to be changed internally as it does not take in seeds as arguments.
    # Please change it in the source code and run this method repeatedly with different seeds. Seeds used: 42, 43, 44
    def unimol(self, rep): 
        performance_dict = {}
        for ss in self.split_strategy:
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            trainer = UniMolTrainer(train_file, test_file)
            save_dir = f'./trained_models/eval_2_scaffold/unimol/rep_{rep}/{ss}'
            trainer.model_training(save_dir, train_batch_size=16, train_epochs=100, early_stopping=10)
            r2, mse, mae = trainer.model_testing(save_dir)
            performance_dict[ss] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open(f'./trained_models/eval_2_scaffold/unimol/rep_{rep}/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
    
    def init_all(self):
        self.lgbm()
        self.lgbm_predictions()
        self.rf()
        self.rf_predictions()
        self.chemberta()
        self.graphormer()
        self.unimol(rep=2)





class ModelAnalyzer:
    def __init__(self):
        self.split_strategy = ['split_1', 'split_2', 'split_3']
        self.embed_extractor = Embeddings()
        self.plot = PlotEmbeddings()
        self.num_features = 40
        self.num_folds = 5
        self.seed = 100
    
    def process_full_data(self): # Adding respective scaffold group to the combined dataset
        data = pd.read_csv('./data/eval_1_combined/dataset/3D_optimizable_dataset.csv')
        with open('./data/eval_2_scaffold/scaffold_idx_list.json', 'r') as f:
            scaffold_idx = json.load(f)
        group_df = pd.DataFrame(index=range(len(data)), columns=['group'])
        for g, index_list in enumerate(scaffold_idx):
            for i in index_list:
                group_df.loc[i,'group'] = g
        full_data = pd.concat([data, group_df], axis=1)
        return full_data
    
    def scaffold_data_prep(self, split_type, full_data, rep_no):
        rf = pd.read_csv(f'./trained_models/eval_2_scaffold/rf/{split_type}/predictions.csv')
        lgbm = pd.read_csv(f'./trained_models/eval_2_scaffold/lightgbm/{split_type}/predictions.csv')
        chemberta = pd.read_csv(f'./trained_models/eval_2_scaffold/chemberta-2/rep_0/{split_type}/predictions.csv')
        graphormer = pd.read_csv(f'./trained_models/eval_2_scaffold/graphormer/{split_type}/predictions.csv')
        unimol = pd.read_csv(f'./trained_models/eval_2_scaffold/unimol/rep_{rep_no}/{split_type}/predictions.csv')
        rf_merged = pd.merge(full_data[['SMILES', 'group']], rf, on='SMILES', how='right')
        lgbm_merged = pd.merge(full_data[['SMILES', 'group']], lgbm, on='SMILES', how='right')
        chemberta_merged = pd.merge(full_data[['SMILES', 'group']], chemberta, on='SMILES', how='right')
        graphormer_merged = pd.merge(full_data[['SMILES', 'group']], graphormer, on='SMILES', how='right')
        unimol_merged = pd.merge(full_data[['SMILES', 'group']], unimol, on='SMILES', how='right')
        # Re-indexing as the merge function to generate descs misaligned the data
        rf_merged = rf_merged.set_index('SMILES').reindex(chemberta_merged['SMILES']).reset_index() 
        lgbm_merged = lgbm_merged.set_index('SMILES').reindex(chemberta_merged['SMILES']).reset_index()
        compiled_predictions = pd.DataFrame()
        compiled_predictions['SMILES'] = unimol_merged['SMILES']
        compiled_predictions['logPapp'] = unimol_merged['logPapp']
        model_names = ['rf', 'lgbm', 'chemberta', 'graphormer', 'unimol']
        prediction_list = [rf_merged, lgbm_merged, chemberta_merged, graphormer_merged, unimol_merged]
        for i, name in enumerate(model_names):
            compiled_predictions[f'{name}_pred'] = prediction_list[i]['pred_logPapp']
        compiled_predictions['group'] = unimol_merged['group']
        sorted_compiled = compiled_predictions.sort_values(by='group', ascending=True)
        return sorted_compiled
    
    def scaffold_distribution(self):
        train_group_freq = []
        test_group_freq = []
        full_data = self.process_full_data()
        for i, ss in enumerate(self.split_strategy):
            test = self.scaffold_data_prep(ss, full_data, rep_no=0) # rep_no here doesn't contribute anything as we are not looking at predictions
            train = full_data[~full_data['SMILES'].isin(test['SMILES'])].sort_values(by='group', ascending=True)
            train_freq = {}
            test_freq = {}
            for g_train in train['group'].unique()[:5]:
                train_freq[g_train] = len(train[train['group'] == g_train])
            for g_test in test['group'].unique()[:5]:
                test_freq[g_test] = len(test[test['group'] == g_test])
            plot_frequency(train_freq, test_freq, split_num=i+1)
            train_group_freq.append(len(train['group'].unique()))
            test_group_freq.append(len(test['group'].unique()))
        plot_diversity(train_group_freq, test_group_freq)
        
    def scaffold_performance(self, rep_no):
        pred_label = ['rf_pred', 'lgbm_pred', 'chemberta_pred', 'graphormer_pred', 'unimol_pred']
        full_data = self.process_full_data()
        for i, ss in enumerate(self.split_strategy):
            split_scaffold = self.scaffold_data_prep(ss, full_data, rep_no)
            for label in pred_label:
                split_scaffold[f'{label}_delta'] = abs(split_scaffold[label] - split_scaffold['logPapp'])
            split_scaffold['group_count'] = split_scaffold.groupby('group')['group'].transform('count')
            uni = split_scaffold.groupby(by='group')['unimol_pred_delta'].mean().sort_index()[:20]
            lgbm = split_scaffold.groupby(by='group')['lgbm_pred_delta'].mean().sort_index()[:20]
            rf = split_scaffold.groupby(by='group')['rf_pred_delta'].mean().sort_index()[:20]
            delta_df = pd.concat([uni, lgbm, rf], axis=1)
            plot_group_performance(delta_df, split_num=i+1, start_grp_idx=0, end_grp_idx=10)
            plot_group_performance(delta_df, split_num=i+1, start_grp_idx=10, end_grp_idx=20)
    
    def shap_init(self, test_desc, trained_model_file):
        X_test = test_desc.drop(columns=['SMILES', 'logPapp'])
        model = joblib.load(trained_model_file)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        return shap_values
    
    def select_top_features(self, test_desc, trained_model_file, num_top_features=100):
        shap_values = self.shap_init(test_desc, trained_model_file)
        X_test = test_desc.drop(columns=['SMILES', 'logPapp'])
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
        top_descs = list(feature_importance.iloc[:num_top_features,0])
        return top_descs
    
    def rf_embeddings(self):
        trainers = ModelTrainer()
        for i, ss in enumerate(self.split_strategy):
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            train_rdkit, test_rdkit = trainers.get_features_rdkit(train_file, test_file)
            for fold in range(self.num_folds):
                rf_model_name = f'RF Split {i+1} Fold {fold}'
                rf_top_descs = self.select_top_features(test_desc=test_rdkit, 
                                                        trained_model_file=f'./trained_models/eval_2_scaffold/rf/{ss}/rdkit_trained_model_fold_{fold}.pkl',
                                                        num_top_features=self.num_features)
                rf_top_descs.append('logPapp')
                self.plot.draw_umap(test_rdkit[rf_top_descs], rf_model_name, 
                                    n_neighbors=200, min_dist=0.5, seed=self.seed)
    
    def lgbm_embeddings(self):
        trainers = ModelTrainer()
        for i, ss in enumerate(self.split_strategy):
            train_file = f'./data/eval_2_scaffold/{ss}_scaffold_train.csv'
            test_file = f'./data/eval_2_scaffold/{ss}_scaffold_test.csv'
            train_rdkit, test_rdkit = trainers.get_features_rdkit(train_file, test_file)
            for fold in range(self.num_folds):
                lgbm_model_name = f'LightGBM Split {i+1} Fold {fold}'
                lgbm_top_descs = self.select_top_features(test_desc=test_rdkit, 
                                                          trained_model_file=f'./trained_models/eval_2_scaffold/lightgbm/{ss}/rdkit_trained_model_fold_{fold}.pkl',
                                                          num_top_features=self.num_features)
                lgbm_top_descs.append('logPapp')
                self.plot.draw_umap(test_rdkit[lgbm_top_descs], lgbm_model_name, 
                                    n_neighbors=200, min_dist=0.5, seed=self.seed)
    
    def cberta_embeddings(self, rep):
        for i, ss in enumerate(self.split_strategy):
            for fold in range(self.num_folds):
                model_name = f'ChemBERTa-2 Split {i+1} Fold {fold}'
                embeddings_df = self.embed_extractor.chemberta(model_dir=f'./trained_models/eval_2_scaffold/chemberta-2/rep_{rep}/{ss}',
                                                               test_dir=f'./data/eval_2_scaffold/{ss}_scaffold_test.csv',
                                                               embeds_dir=f'./embeddings/eval_2_scaffold/chemberta-2/{ss}',
                                                               fold_no=fold)
                self.plot.draw_umap(embeddings_df, model_name, 
                                    n_neighbors=200, min_dist=0.5, seed=self.seed)
    
    def graphormer_embeddings(self):
        for i, ss in enumerate(self.split_strategy):
            for fold in range(self.num_folds):
                model_name = f'Graphormer Split {i+1} Fold {fold}'
                embeddings_df = self.embed_extractor.graphormer(model_dir=f'./trained_models/eval_2_scaffold/graphormer/{ss}',
                                                                test_dir=f'./data/2D_graph_data/eval_2_scaffold/{ss}',
                                                                embeds_dir=f'./embeddings/eval_2_scaffold/graphormer/{ss}',
                                                                fold_no=fold)
                self.plot.draw_umap(embeddings_df, model_name, 
                                    n_neighbors=200, min_dist=0.5, seed=self.seed)
    
    def unimol_embeddings(self, rep):
        for i, ss in enumerate(self.split_strategy):
            for fold in range(self.num_folds):
                model_name = f'Uni-Mol Split {i+1} Fold {fold}'
                embeddings_df = self.embed_extractor.unimol(model_dir=f'./trained_models/eval_2_scaffold/unimol/rep_{rep}/{ss}',
                                                            test_dir=f'./data/eval_2_scaffold/{ss}_scaffold_test.csv',
                                                            embeds_dir=f'./embeddings/eval_2_scaffold/unimol/{ss}',
                                                            fold_no=fold)
                self.plot.draw_umap(embeddings_df, model_name, 
                                    n_neighbors=200, min_dist=0.5, seed=self.seed)
    
    def init_all(self):
        self.scaffold_distribution()
        self.scaffold_performance(rep_no=0)
        self.scaffold_performance(rep_no=1)
        self.scaffold_performance(rep_no=2)
        self.rf_embeddings()
        self.lgbm_embeddings()
        self.cberta_embeddings(rep=0)
        self.graphormer_embeddings()
        self.unimol_embeddings(rep=0)
    
    
    
    
    
class Helper:
    def compile_fold_predictions_cberta(self):
        n_splits = 3
        num_folds = 5
        n_repetitions = 3
        for rep in range(n_repetitions):
            for split in range(n_splits):
                avg_pred = pd.DataFrame()
                for fold in range(num_folds):
                    pred = pd.read_csv(f'./trained_models/eval_2_scaffold/chemberta-2/rep_{rep}/split_{split+1}/fold_{fold}/predictions.csv')
                    avg_pred[f'fold_{fold}_pred'] = pred['pred_logPapp']
                avg_pred['pred_logPapp'] = avg_pred.mean(axis=1)
                avg_pred['SMILES'] = pred['SMILES']
                avg_pred['logPapp'] = pred['logPapp']
                avg_pred.to_csv(f'./trained_models/eval_2_scaffold/chemberta-2/rep_{rep}/split_{split+1}/predictions.csv', index=False)
    
    def compile_fold_predictions_graphormer(self):
        n_splits = 3
        num_folds = 5
        for split in range(n_splits):
            avg_pred = pd.DataFrame()
            for fold in range(num_folds):
                pred = pd.read_csv(f'./trained_models/eval_2_scaffold/graphormer/split_{split+1}/fold_{fold}/predictions.csv')
                avg_pred[f'fold_{fold}_pred'] = pred['pred_logPapp']
            avg_pred['pred_logPapp'] = avg_pred.mean(axis=1)
            avg_pred['SMILES'] = pred['SMILES']
            avg_pred['logPapp'] = pred['logPapp']
            avg_pred.to_csv(f'./trained_models/eval_2_scaffold/graphormer/split_{split+1}/predictions.csv', index=False)
    
    def compile_fold_predictions(self):
        self.compile_fold_predictions_cberta()
        self.compile_fold_predictions_graphormer()
        
def main():
    splitter = ScaffoldSplit()
    trainers = ModelTrainer()
    analyzer = ModelAnalyzer()
    helper = Helper()
    splitter.scaffold_split()
    trainers.init_all()
    helper.compile_fold_predictions()
    analyzer.init_all()

if __name__ == "__main__":

    main()
