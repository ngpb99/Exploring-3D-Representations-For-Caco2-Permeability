import pandas as pd
from engine import PredefinedFeatures
from unimol_tools import MolPredict
import joblib

class TransporterEval:
    def __init__(self):
        self.transport_file = './data/eval_4_transporter/dataset/transporter_dataset.csv'
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
    
    def init_all(self):
        self.lgbm_combined_eval()
        self.lgbm_lit_eval()
        self.unimol_combined_eval()
        self.unimol_literature_eval()

def main():
    evaluate = TransporterEval()
    evaluate.init_all()

if __name__ == "__main__":
    main()