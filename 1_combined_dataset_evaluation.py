from engine import PredefinedFeatures, GraphData, LGBMTrainer, RFTrainer, CBERTaTrainer, GraphormerTrainer, UniMolTrainer
from visualization import PlotEmbeddings, plot_predictions, draw_mols, plot_top_feature_performance
from embeddings import Embeddings
from rdkit.Chem import rdFingerprintGenerator, Draw
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from IPython.display import display
import pandas as pd
import numpy as np
import joblib
import shap
import json

class ModelTrainer:
    def __init__(self):
        self.data_file = './data/eval_1_combined/dataset/3D_optimizable_dataset.csv'
        self.train_file = './data/eval_1_combined/dataset/train_data.csv'
        self.test_file = './data/eval_1_combined/dataset/test_data.csv'
        self.n_repetition = 3
        self.num_folds = 5
    
    def get_features(self):
        data = pd.read_csv(self.data_file)
        train = pd.read_csv(self.train_file)
        test = pd.read_csv(self.test_file)
        save_dir = './data/eval_1_combined/predefined_descs'
        calc = PredefinedFeatures(data, train, test, save_dir)
        train_rdkit, test_rdkit = calc.rdkit_descs(train_file_name='rdkit_train.csv', test_file_name='rdkit_test.csv')
        train_cfps, test_cfps = calc.morgan_cfps(train_file_name='countfp_train.csv', test_file_name='countfp_test.csv')
        train_bitfps, test_bitfps = calc.morgan_bitfps(train_file_name='bitfp_train.csv', test_file_name='bitfp_test.csv')
        train_mordred, test_mordred = calc.mordred_descs(train_file_name='mordred_train.csv', test_file_name='mordred_test.csv')
        train_list = [train_rdkit, train_cfps, train_bitfps, train_mordred]
        test_list = [test_rdkit, test_cfps, test_bitfps, test_mordred]
        return train_list, test_list
    
    def generate_graph(self):
        train = pd.read_csv(self.train_file)
        test = pd.read_csv(self.test_file)
        generator = GraphData()
        generator.save_graph(train, test, graph_dir='./data/2D_graph_data/eval_1_combined')
    
    def lgbm(self):
        train_list, test_list = self.get_features()
        names = ['rdkit', 'cfps', 'bitfps', 'mordred']
        params_dict = {}
        performance_dict = {}
        for i in range(len(train_list)):
            mae_score = []
            mse_score = []
            r2_score = []
            for rep in range(self.n_repetition):
                model_dir = f'./trained_models/eval_1_combined/lightgbm/rep_{rep}'
                trainer = LGBMTrainer(train_list[i], test_list[i], seed=rep)
                best_params = trainer.model_tuning()
                params_dict[f'{names[i]}_rep_{rep}'] = best_params
                r2, mse, mae = trainer.model_testing(names[i], best_params, model_dir)
                r2_score.append(r2)
                mse_score.append(mse)
                mae_score.append(mae)
            performance_dict[names[i]] = (
                f'R2: {np.mean(r2_score):.4f} +/- {np.std(r2_score):.4f} '
                f'MAE: {np.mean(mae_score):.4f} +/- {np.std(mae_score):.4f} '
                f'MSE: {np.mean(mse_score):.4f} +/- {np.std(mse_score):.4f}'
            )
        with open('./trained_models/eval_1_combined/lightgbm/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
        with open('./trained_models/eval_1_combined/lightgbm/best_hyperparams.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    def lgbm_predictions(self, desc_name):
        train = pd.read_csv(f'./data/eval_1_combined/predefined_descs/{desc_name}_train.csv')
        test = pd.read_csv(f'./data/eval_1_combined/predefined_descs/{desc_name}_test.csv')
        for rep in range(self.n_repetition):
            trainer = LGBMTrainer(train, test, seed=rep)
            predictions_df = pd.DataFrame()
            for fold in range(self.num_folds):
                model_file = f'./trained_models/eval_1_combined/lightgbm/rep_{rep}/{desc_name}_trained_model_fold_{fold}.pkl'
                pred = trainer.model_predict(model_file)
                predictions_df[f'fold_{fold}_pred'] = pred
            predictions_df['pred_logPapp'] = predictions_df.mean(axis=1)
            predictions_df['SMILES'] = test['SMILES']
            predictions_df['logPapp'] = test['logPapp']
            predictions_df.to_csv(f'./trained_models/eval_1_combined/lightgbm/rep_{rep}/predictions.csv', index=False)
    
    def rf(self):
        train_list, test_list = self.get_features()
        names = ['rdkit', 'cfps', 'bitfps', 'mordred']
        params_dict = {}
        performance_dict = {}
        for i in range(len(train_list)):
            mae_score = []
            mse_score = []
            r2_score = []
            for rep in range(self.n_repetition):
                model_dir = f'./trained_models/eval_1_combined/rf/rep_{rep}'
                trainer = RFTrainer(train_list[i], test_list[i], seed=rep)
                best_params = trainer.model_tuning()
                params_dict[f'{names[i]}_rep_{rep}'] = best_params
                r2, mse, mae = trainer.model_testing(names[i], best_params, model_dir)
                r2_score.append(r2)
                mse_score.append(mse)
                mae_score.append(mae)
            performance_dict[names[i]] = (
                f'R2: {np.mean(r2_score):.4f} +/- {np.std(r2_score):.4f} '
                f'MAE: {np.mean(mae_score):.4f} +/- {np.std(mae_score):.4f} '
                f'MSE: {np.mean(mse_score):.4f} +/- {np.std(mse_score):.4f}'
            )
        with open('./trained_models/eval_1_combined/rf/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
        with open('./trained_models/eval_1_combined/rf/best_hyperparams.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    def rf_predictions(self, desc_name):
        train = pd.read_csv(f'./data/eval_1_combined/predefined_descs/{desc_name}_train.csv')
        test = pd.read_csv(f'./data/eval_1_combined/predefined_descs/{desc_name}_test.csv')
        for rep in range(self.n_repetition):
            trainer = RFTrainer(train, test, seed=rep)
            predictions_df = pd.DataFrame()
            for fold in range(self.num_folds):
                model_file = f'./trained_models/eval_1_combined/rf/rep_{rep}/{desc_name}_trained_model_fold_{fold}.pkl'
                pred = trainer.model_predict(model_file)
                predictions_df[f'fold_{fold}_pred'] = pred
            predictions_df['pred_logPapp'] = predictions_df.mean(axis=1)
            predictions_df['SMILES'] = test['SMILES']
            predictions_df['logPapp'] = test['logPapp']
            predictions_df.to_csv(f'./trained_models/eval_1_combined/rf/rep_{rep}/predictions.csv', index=False)
    
    def chemberta(self):
        seed_list = [123, 124, 125]
        train = pd.read_csv(self.train_file)
        test = pd.read_csv(self.test_file)
        trainer = CBERTaTrainer(train, test, model_name='DeepChem/ChemBERTa-77M-MTR')
        mae_score = []
        mse_score = []
        r2_score = []
        performance_dict = {}
        for rep in range(self.n_repetition):
            save_dir = f'./trained_models/eval_1_combined/chemberta-2/rep_{rep}'
            trainer.model_training(save_dir, train_batch_size=128, eval_batch_size=128, train_epoch=100, 
                                   eval_steps=10, max_length=195, seed=seed_list[rep])
            r2, mse, mae = trainer.model_testing(save_dir, max_length=195)
            r2_score.append(r2)
            mse_score.append(mse)
            mae_score.append(mae)
        performance_dict['chemberta-2 performance'] = (
            f'R2: {np.mean(r2_score):.4f} +/- {np.std(r2_score):.4f} '
            f'MAE: {np.mean(mae_score):.4f} +/- {np.std(mae_score):.4f} '
            f'MSE: {np.mean(mse_score):.4f} +/- {np.std(mse_score):.4f}'
        )
        with open('./trained_models/eval_1_combined/chemberta-2/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
    
    def graphormer(self):
        seed_list = [123, 124, 125]
        self.generate_graph()
        trainer = GraphormerTrainer(data_dir='./data/2D_graph_data/eval_1_combined', model_name='clefourrier/graphormer-base-pcqm4mv2')
        mae_score = []
        mse_score = []
        r2_score = []
        performance_dict = {}
        for rep in range(self.n_repetition):
            save_dir = f'./trained_models/eval_1_combined/graphormer/rep_{rep}'
            trainer.model_training(save_dir, train_batch_size=128, eval_batch_size=128, train_epoch=100, 
                                   eval_steps=10, seed=seed_list[rep])
            r2, mse, mae = trainer.model_testing(save_dir, test_file=self.test_file)
            r2_score.append(r2)
            mse_score.append(mse)
            mae_score.append(mae)
        performance_dict['graphormer performance'] = (
            f'R2: {np.mean(r2_score):.4f} +/- {np.std(r2_score):.4f} '
            f'MAE: {np.mean(mae_score):.4f} +/- {np.std(mae_score):.4f} '
            f'MSE: {np.mean(mse_score):.4f} +/- {np.std(mse_score):.4f}'
        )
        with open('./trained_models/eval_1_combined/graphormer/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)

    # Seed for Uni-Mol needs to be changed internally as it does not take in seeds as arguments.
    # Please change it in the source code and run this method repeatedly with different seeds. Seeds used: 42, 43, 44
    def unimol(self, rep): 
        trainer = UniMolTrainer(self.train_file, self.test_file)
        save_dir = f'./trained_models/eval_1_combined/unimol/rep_{rep}'
        trainer.model_training(save_dir, train_batch_size=16, train_epochs=100, early_stopping=10)

    def unimol_eval(self):
        trainer = UniMolTrainer(self.train_file, self.test_file)
        mae_score = []
        mse_score = []
        r2_score = []
        performance_dict = {}
        for rep in range(self.n_repetition):
            save_dir = f'./trained_models/eval_1_combined/unimol/rep_{rep}'
            r2, mse, mae = trainer.model_testing(save_dir)
            r2_score.append(r2)
            mse_score.append(mse)
            mae_score.append(mae)
        performance_dict['unimol performance'] = (
            f'R2: {np.mean(r2_score):.4f} +/- {np.std(r2_score):.4f} '
            f'MAE: {np.mean(mae_score):.4f} +/- {np.std(mae_score):.4f} '
            f'MSE: {np.mean(mse_score):.4f} +/- {np.std(mse_score):.4f}'
        )
        with open('./trained_models/eval_1_combined/unimol/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
    
    def init_all(self):
        self.lgbm()
        self.lgbm_predictions(desc_name='rdkit') # Predict for RDKit only as it is best descriptor
        self.rf()
        self.rf_predictions(desc_name='rdkit')
        self.chemberta()
        self.graphormer()
        self.unimol(rep=2)
        self.unimol_eval()





class ModelAnalyzer:
    def __init__(self):
        self.n_neighbors = [2, 5, 10, 20, 50, 100, 200]
        self.min_dist = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
        self.plot = PlotEmbeddings()
        self.embed_extractor = Embeddings()
        self.n_repetition = 3
        self.num_folds = 5
        self.seed = 100
    
    def shap_init(self, test_desc_file, trained_model_file):
        test = pd.read_csv(test_desc_file)
        X_test = test.drop(columns=['SMILES', 'logPapp'])
        model = joblib.load(trained_model_file)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        return shap_values
        
    def shap_plot(self, test_desc_file, model_name, desc_type):
        if desc_type == 'rdkit':
            for fold in range(self.num_folds): # Not looping through repetition as first rep will give us necessary insights
                trained_model_file = f'./trained_models/eval_1_combined/{model_name}/rep_0/rdkit_trained_model_fold_{fold}.pkl'
                shap_values = self.shap_init(test_desc_file, trained_model_file)
                shap.plots.bar(shap_values, max_display=9)
                plt.show()
                shap.summary_plot(shap_values, max_display=9)
                plt.show()
        elif desc_type == 'mordred':
            for fold in range(self.num_folds): # Not looping through repetition as first rep will give us necessary insights
                trained_model_file = f'./trained_models/eval_1_combined/{model_name}/rep_0/mordred_trained_model_fold_{fold}.pkl'
                shap_values = self.shap_init(test_desc_file, trained_model_file)
                shap.plots.bar(shap_values, max_display=9)
                plt.show()
                shap.summary_plot(shap_values, max_display=9)
                plt.show()
        
    def select_top_features(self, test_desc_file, trained_model_file, num_top_features=100):
        shap_values = self.shap_init(test_desc_file, trained_model_file)
        test_rdkit = pd.read_csv(test_desc_file)
        X_test = test_rdkit.drop(columns=['SMILES', 'logPapp'])
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
        top_descs = list(feature_importance.iloc[:num_top_features,0])
        return top_descs
    
    def lgbm_top_descs_analysis(self, test_desc_file, train_desc_file, num_top_features=100):
        for rep in range(self.n_repetition):
            for fold in range(self.num_folds):
                trained_model_file=f'./trained_models/eval_1_combined/lightgbm/rep_{rep}/rdkit_trained_model_fold_{fold}.pkl'
                train_rdkit = pd.read_csv(train_desc_file)
                test_rdkit = pd.read_csv(test_desc_file)
                top_descs = self.select_top_features(test_desc_file, trained_model_file)
                with open('./trained_models/eval_1_combined/lightgbm/best_hyperparams.json', 'r') as f:
                    hyperparams_dict = json.load(f)
                best_params = hyperparams_dict[f'rdkit_rep_{rep}']
                r2_results = {}
                mae_results = {}
                mse_results = {}
                for i in range(1, num_top_features+1):
                    features_to_include = top_descs[:i]
                    features_to_include.extend(['SMILES','logPapp'])
                    trainer = LGBMTrainer(train_rdkit[features_to_include], test_rdkit[features_to_include], seed=0)
                    r2, mse, mae = trainer.top_desc_training(best_params, trained_model_file)
                    r2_results[f'top_{i}_feature'] = r2
                    mae_results[f'top_{i}_feature'] = mae
                    mse_results[f'top_{i}_feature'] = mse
                plot_top_feature_performance(mae_results, model_name='LightGBM')
    
    def rf_top_descs_analysis(self, test_desc_file, train_desc_file, num_top_features=100):
        for rep in range(self.n_repetition):
            for fold in range(self.num_folds):
                trained_model_file=f'./trained_models/eval_1_combined/rf/rep_{rep}/rdkit_trained_model_fold_{fold}.pkl'
                train_rdkit = pd.read_csv(train_desc_file)
                test_rdkit = pd.read_csv(test_desc_file)
                top_descs = self.select_top_features(test_desc_file, trained_model_file)
                with open('./trained_models/eval_1_combined/rf/best_hyperparams.json', 'r') as f:
                    hyperparams_dict = json.load(f)
                best_params = hyperparams_dict[f'rdkit_rep_{rep}']
                r2_results = {}
                mae_results = {}
                mse_results = {}
                for i in range(1, num_top_features+1):
                    features_to_include = top_descs[:i]
                    features_to_include.extend(['SMILES','logPapp'])
                    trainer = RFTrainer(train_rdkit[features_to_include], test_rdkit[features_to_include], seed=0)
                    r2, mse, mae = trainer.top_desc_training(best_params, trained_model_file)
                    r2_results[f'top_{i}_feature'] = r2
                    mae_results[f'top_{i}_feature'] = mae
                    mse_results[f'top_{i}_feature'] = mse
                plot_top_feature_performance(mae_results, model_name='RF')
        
    def prediction_plot(self):
        model_names = ['lightgbm', 'rf', 'chemberta-2', 'graphormer', 'unimol']
        plot_titles = ['LightGBM Performance w/ RDKit', 'RF Performance w/ RDKit', 'ChemBERTa-2 Performance', 'Graphormer Performance', 'Uni-Mol Performance']
        for rep in range(self.n_repetition):
            for idx, name in enumerate(model_names):
                prediction_file = f'./trained_models/eval_1_combined/{name}/rep_{rep}/predictions.csv'
                pred = pd.read_csv(prediction_file)
                pred['group'] = pd.cut(pred['logPapp'], bins=[-float('inf'), 0, 1, float('inf')], labels=[1,2,3], right=False)
                plot_predictions(pred, plot_titles[idx], x_label='logPapp', y_label='pred_logPapp')
    
    def poorest_performing_mol(self, df):
        df['delta'] = abs(df['logPapp'] - df['pred_logPapp'])
        return df.sort_values(by=['delta'], ascending=False).iloc[:20, :]['SMILES']
    
    def poor_performance_eval(self, rep_no, draw=True):
        lgbm_pred = pd.read_csv(f'./trained_models/eval_1_combined/lightgbm/rep_{rep_no}/predictions.csv')
        rf_pred = pd.read_csv(f'./trained_models/eval_1_combined/rf/rep_{rep_no}/predictions.csv')
        cberta_pred = pd.read_csv(f'./trained_models/eval_1_combined/chemberta-2/rep_{rep_no}/predictions.csv')
        graphormer_pred = pd.read_csv(f'./trained_models/eval_1_combined/graphormer/rep_{rep_no}/predictions.csv')
        unimol_pred = pd.read_csv(f'./trained_models/eval_1_combined/unimol/rep_{rep_no}/predictions.csv')
        model_names = ['lightgbm', 'rf', 'chemberta-2', 'graphormer', 'unimol']
        pred_list = [lgbm_pred, rf_pred, cberta_pred, graphormer_pred, unimol_pred]
        poorest_performing = pd.DataFrame()
        for idx, name in enumerate(model_names):
            poorest_performing[name] = self.poorest_performing_mol(pred_list[idx])
        poorest_performing_all = poorest_performing.dropna(axis=0)
        poorest_performing_all = pd.DataFrame(poorest_performing_all.iloc[:, 0]).rename(columns={'lightgbm': 'SMILES'})
        if draw:
            poorest_performing_all['logPapp'] = lgbm_pred['logPapp']
            m = [Chem.MolFromSmiles(s) for s in poorest_performing_all['SMILES']]
            legends = [f"True: {t:.3f}\nMolecule {i}" for t, i in zip(poorest_performing_all['logPapp'], ['A', 'B', 'C', 'D', 'E', 'F'])]
            draw_mols(legends, mols=m[:6], mols_per_row=3)
        return poorest_performing_all['SMILES'][:6]
    
    def execute_poor_eval_loop(self):
        for rep in range(self.n_repetition):
            poor_mols = self.poor_performance_eval(rep_no=rep, draw=True)
    
    def similarity_analysis(self):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        train_data = pd.read_csv('./data/eval_1_combined/dataset/train_data.csv')
        target_names = ['A', 'B', 'C', 'D', 'E', 'F']
        true_values = pd.DataFrame(columns=['mol_1', 'mol_2', 'mol_3', 'mol_4'])
        compiled_dict = {}
        all_pred = self.compile_poorest_predictions()
        for rep in range(self.n_repetition): # Calculating the most similar train molecules to poorest performing molecules across 3 reps
            similarity_dict = {}
            poorest_performing = self.poor_performance_eval(rep)
            for idx, mol_idx in enumerate(poorest_performing.index):
                score_dict = {}
                target_smiles = list(poorest_performing)[idx]
                target_cfp = mfpgen.GetCountFingerprint(Chem.MolFromSmiles(target_smiles))
                all_smiles = train_data['SMILES']
                ms = [Chem.MolFromSmiles(s) for s in all_smiles]
                cfps = [mfpgen.GetCountFingerprint(x) for x in ms]
                for i, cfp in enumerate(cfps):
                    sim = DataStructs.TanimotoSimilarity(target_cfp, cfp)
                    score_dict[all_smiles[i]] = sim
                sorted_score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
                sim_list = []
                for number in range(4):
                    sim_list.append(list(sorted_score_dict.keys())[number])
                similarity_dict[target_smiles] = sim_list
            for idx, (key, value) in enumerate(similarity_dict.items()): # Drawing the molecules out for visualization
                mol = Chem.MolFromSmiles(key)
                img = Draw.MolToImage(mol, legend=f'Molecule {target_names[idx]}')
                display(img)
                mols = [Chem.MolFromSmiles(s) for s in value]
                legends = [f"Mol {i}" for i in range(1,5)]
                draw_mols(legends, mols=mols, mols_per_row=2)
                smiles_to_logpapp = dict(zip(train_data['SMILES'], train_data['logPapp']))
                train_logpapp = [smiles_to_logpapp[smi] for smi in value]
                true_values.loc[f'rep_{rep}_Mol_{target_names[idx]}'] = train_logpapp
            compiled_dict[f'rep_{rep}'] = similarity_dict # Keeping a copy of the SMILES most similar to poorest performing molecules in each rep
        true_values = true_values.reset_index()
        true_values['mean_pred'], true_values['std'] = all_pred['mean_pred'], all_pred['std']
        # Contains logPapp of most similar train molecules and predicted test molecules to show influence these molecules on the test molecule.
        true_values.to_csv('./prediction_analysis/eval_1_results/poorest_test_pred_vs_similar_training_true.csv', index=False) 
        with open('./prediction_analysis/eval_1_results/poorest_mols_similarity_analysis.json', 'w') as f:
            json.dump(compiled_dict, f, indent=2)
            
    def compile_poorest_predictions(self):
        total = pd.DataFrame(columns=['lgbm', 'rf', 'cberta', 'graphormer', 'unimol'])
        for rep in range(self.n_repetition):
            prediction_value = pd.DataFrame()
            lgbm_pred = pd.read_csv(f'./trained_models/eval_1_combined/lightgbm/rep_{rep}/predictions.csv')
            rf_pred = pd.read_csv(f'./trained_models/eval_1_combined/rf/rep_{rep}/predictions.csv')
            cberta_pred = pd.read_csv(f'./trained_models/eval_1_combined/chemberta-2/rep_{rep}/predictions.csv')
            graphormer_pred = pd.read_csv(f'./trained_models/eval_1_combined/graphormer/rep_{rep}/predictions.csv')
            unimol_pred = pd.read_csv(f'./trained_models/eval_1_combined/unimol/rep_{rep}/predictions.csv')
            poorest_performing = self.poor_performance_eval(rep, draw=False)
            prediction_value['lgbm'] = lgbm_pred.loc[poorest_performing.index,'pred_logPapp']
            prediction_value['rf'] = rf_pred.loc[poorest_performing.index,'pred_logPapp']
            prediction_value['cberta'] = cberta_pred.loc[poorest_performing.index,'pred_logPapp']
            prediction_value['graphormer'] = graphormer_pred.loc[poorest_performing.index,'pred_logPapp']
            prediction_value['unimol'] = unimol_pred.loc[poorest_performing.index,'pred_logPapp']
            total = pd.concat([total, prediction_value], axis=0)
        total['mean_pred'] = total.iloc[:,:5].mean(axis=1)
        total['std'] = total.iloc[:,:5].std(axis=1)
        total = total.reset_index(drop=True)
        return total
    
    def lgbm_embeddings(self, num_features): # Num features determined from Predictions vs Num top features plot
        test_desc_file = './data/eval_1_combined/predefined_descs/rdkit_test.csv'
        embeddings_df = pd.read_csv(test_desc_file)
        # Using rep 0 fold 0 model to select optimal neighbours and min distance
        top_descs = self.select_top_features(test_desc_file=test_desc_file, 
                                             trained_model_file='./trained_models/eval_1_combined/lightgbm/rep_0/rdkit_trained_model_fold_0.pkl',
                                             num_top_features=num_features)
        top_descs.append('logPapp')
        model_name = 'Top RDKit Descs for LightGBM'
        for neighbours in self.n_neighbors:
            for dist in self.min_dist:
                self.plot.draw_umap(embeddings_df[top_descs], model_name, n_neighbors=neighbours, min_dist=dist, seed=self.seed)
    
    def rf_embeddings(self, num_features):
        test_desc_file = './data/eval_1_combined/predefined_descs/rdkit_test.csv'
        embeddings_df = pd.read_csv(test_desc_file)
        top_descs = self.select_top_features(test_desc_file=test_desc_file, 
                                             trained_model_file='./trained_models/eval_1_combined/lightgbm/rep_0/rdkit_trained_model_fold_0.pkl',
                                             num_top_features=num_features)
        top_descs.append('logPapp')
        model_name = 'Top RDKit Descs for RF'
        for neighbours in self.n_neighbors:
            for dist in self.min_dist:
                self.plot.draw_umap(embeddings_df[top_descs], model_name, n_neighbors=neighbours, min_dist=dist, seed=self.seed)
    
    def cberta_embeddings(self):
        embeddings_df = self.embed_extractor.chemberta(model_dir='./trained_models/eval_1_combined/chemberta-2/rep_0',
                                                       test_dir='./data/eval_1_combined/dataset/test_data.csv',
                                                       embeds_dir='./embeddings/eval_1_combined/chemberta-2/rep_0',
                                                       fold_no=0)
        model_name = 'ChemBERTa-2'
        for neighbours in self.n_neighbors:
            for dist in self.min_dist:
                self.plot.draw_umap(embeddings_df, model_name, n_neighbors=neighbours, min_dist=dist, seed=self.seed)
    
    def graphormer_embeddings(self):
        embeddings_df = self.embed_extractor.graphormer(model_dir='./trained_models/eval_1_combined/graphormer/rep_0', 
                                                        test_dir='./data/2D_graph_data/eval_1_combined', 
                                                        embeds_dir='./embeddings/eval_1_combined/graphormer/rep_0', 
                                                        fold_no=0)
        model_name = 'Graphormer'
        for neighbours in self.n_neighbors:
            for dist in self.min_dist:
                self.plot.draw_umap(embeddings_df, model_name, n_neighbors=neighbours, min_dist=dist, seed=self.seed)
    
    def unimol_embeddings(self):
        embeddings_df = self.embed_extractor.unimol(model_dir='./trained_models/eval_1_combined/unimol/rep_0', 
                                                    test_dir='./data/eval_1_combined/dataset/test_data.csv', 
                                                    embeds_dir='./embeddings/eval_1_combined/unimol/rep_0', 
                                                    fold_no=0)
        model_name = 'Uni-Mol'
        for neighbours in self.n_neighbors:
            for dist in self.min_dist:
                self.plot.draw_umap(embeddings_df, model_name, n_neighbors=neighbours, min_dist=dist, seed=self.seed)

    def embeddings_plot_all(self):
        test_desc_file = './data/eval_1_combined/predefined_descs/rdkit_test.csv'
        lgbm_rf_embeds = pd.read_csv(test_desc_file)
        test_dir = './data/eval_1_combined/dataset/test_data.csv'
        num_features = 40
        neighbours = 200
        dist = 0.5
        for rep in range(self.n_repetition):
            for fold in range(self.num_folds):
                lgbm_model_name = f'LightGBM Rep {rep} Fold {fold}'
                rf_model_name = f'RF Rep {rep} Fold {fold}'
                cberta_model_name = f'ChemBERTa-2 Rep {rep} Fold {fold}'
                graphormer_model_name = f'Graphormer Rep {rep} Fold {fold}'
                unimol_model_name = f'Uni-Mol Rep {rep} Fold {fold}'
                lgbm_top_descs = self.select_top_features(test_desc_file=test_desc_file, 
                                                          trained_model_file=f'./trained_models/eval_1_combined/lightgbm/rep_{rep}/rdkit_trained_model_fold_{fold}.pkl',
                                                          num_top_features=num_features)
                lgbm_top_descs.append('logPapp')
                rf_top_descs = self.select_top_features(test_desc_file=test_desc_file, 
                                                        trained_model_file=f'./trained_models/eval_1_combined/rf/rep_{rep}/rdkit_trained_model_fold_{fold}.pkl',
                                                        num_top_features=num_features)
                rf_top_descs.append('logPapp')
                cberta_embeds = self.embed_extractor.chemberta(model_dir=f'./trained_models/eval_1_combined/chemberta-2/rep_{rep}',
                                                               test_dir=test_dir,
                                                               embeds_dir=f'./embeddings/eval_1_combined/chemberta-2/rep_{rep}',
                                                               fold_no=fold)
                graphormer_embeds = self.embed_extractor.graphormer(model_dir=f'./trained_models/eval_1_combined/graphormer/rep_{rep}',
                                                                    test_dir='./data/2D_graph_data/eval_1_combined',
                                                                    embeds_dir=f'./embeddings/eval_1_combined/graphormer/rep_{rep}',
                                                                    fold_no=fold)
                unimol_embeds = self.embed_extractor.unimol(model_dir=f'./trained_models/eval_1_combined/unimol/rep_{rep}',
                                                            test_dir=test_dir,
                                                            embeds_dir=f'./embeddings/eval_1_combined/unimol/rep_{rep}',
                                                            fold_no=fold)
                self.plot.draw_umap(lgbm_rf_embeds[lgbm_top_descs], lgbm_model_name, 
                                    n_neighbors=neighbours, min_dist=dist, seed=self.seed)
                self.plot.draw_umap(lgbm_rf_embeds[rf_top_descs], rf_model_name, 
                                    n_neighbors=neighbours, min_dist=dist, seed=self.seed)
                self.plot.draw_umap(cberta_embeds, cberta_model_name, 
                                    n_neighbors=neighbours, min_dist=dist, seed=self.seed)
                self.plot.draw_umap(graphormer_embeds, graphormer_model_name, 
                                    n_neighbors=neighbours, min_dist=dist, seed=self.seed)
                self.plot.draw_umap(unimol_embeds, unimol_model_name, 
                                    n_neighbors=neighbours, min_dist=dist, seed=self.seed)
                
    def init_all(self):
        test_rdkit_file = './data/eval_1_combined/predefined_descs/rdkit_test.csv'
        test_mordred_file = './data/eval_1_combined/predefined_descs/mordred_test.csv'
        train_desc_file = './data/eval_1_combined/predefined_descs/rdkit_train.csv'
        self.shap_plot(test_rdkit_file, model_name='lightgbm', desc_type='rdkit')
        self.shap_plot(test_rdkit_file, model_name='rf', desc_type='rdkit')
        self.shap_plot(test_mordred_file, model_name='lightgbm', desc_type='mordred')
        self.shap_plot(test_mordred_file, model_name='rf', desc_type='mordred')
        self.lgbm_top_descs_analysis(test_rdkit_file, train_desc_file)
        self.rf_top_descs_analysis(test_rdkit_file, train_desc_file)
        self.prediction_plot()
        self.execute_poor_eval_loop()
        self.similarity_analysis()
        self.lgbm_embeddings(num_features=40)
        self.rf_embeddings(num_features=40)
        self.graphormer_embeddings()
        self.unimol_embeddings()
        self.embeddings_plot_all()
                




class Helper:
    def compile_fold_predictions(self):
        n_repetition = 3
        num_folds = 5
        model_names = ['chemberta-2', 'graphormer']
        for n in model_names:
            for rep in range(n_repetition):
                avg_pred = pd.DataFrame()
                for fold in range(num_folds):
                    pred = pd.read_csv(f'./trained_models/eval_1_combined/{n}/rep_{rep}/fold_{fold}/predictions.csv')
                    avg_pred[f'fold_{fold}_pred'] = pred['pred_logPapp']
                avg_pred['pred_logPapp'] = avg_pred.mean(axis=1)
                avg_pred['SMILES'] = pred['SMILES']
                avg_pred['logPapp'] = pred['logPapp']
                avg_pred.to_csv(f'./trained_models/eval_1_combined/{n}/rep_{rep}/predictions.csv', index=False)

def main():
    trainers = ModelTrainer()
    analyzer = ModelAnalyzer()
    helper = Helper()
    trainers.init_all()
    helper.compile_fold_predictions()
    analyzer.init_all()

if __name__ == "__main__":

    main()

