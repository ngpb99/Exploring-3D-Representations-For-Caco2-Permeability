from engine import PredefinedFeatures, GraphData, LGBMTrainer, RFTrainer, CBERTaTrainer, GraphormerTrainer, UniMolTrainer
import pandas as pd
import json

class ModelTrainer:
    def __init__(self):
        self.lit_names = ['wang_2016', 'wang_2020', 'wang_chen', 'pytdc']
        self.seed1 = 0
        self.seed2 = 123
        self.n_repetition = 3
    
    def get_features(self, train_file, test_file, lit_name):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        data = pd.concat([train, test], axis=0).reset_index(drop=True)
        save_dir = './data/eval_3_literature/predefined_descs'
        calc = PredefinedFeatures(data, train, test, save_dir)
        train_rdkit, test_rdkit = calc.rdkit_descs(train_file_name=f'{lit_name}_rdkit_train.csv', test_file_name=f'{lit_name}_rdkit_test.csv')
        return train_rdkit, test_rdkit
    
    def generate_graph(self, train_file, test_file, lit_name):
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        generator = GraphData()
        generator.save_graph(train, test, graph_dir=f'./data/2D_graph_data/eval_3_literature/{lit_name}')
    
    def lgbm(self):
        params_dict = {}
        performance_dict = {}
        for name in self.lit_names:
            train_file = f'./data/eval_3_literature/dataset/{name}_train.csv'
            test_file = f'./data/eval_3_literature/dataset/{name}_test.csv'
            train_rdkit, test_rdkit = self.get_features(train_file, test_file, lit_name=name)
            model_dir = f'./trained_models/eval_3_literature/lightgbm/{name}'
            trainer = LGBMTrainer(train_rdkit, test_rdkit, seed=self.seed1)
            best_params = trainer.model_tuning()
            params_dict[name] = best_params
            r2, mse, mae = trainer.testing_literature('rdkit', best_params, model_dir)
            performance_dict[name] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open('./trained_models/eval_3_literature/lightgbm/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
        with open('./trained_models/eval_3_literature/lightgbm/best_hyperparams.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    def rf(self):
        params_dict = {}
        performance_dict = {}
        for name in self.lit_names:
            train_file = f'./data/eval_3_literature/dataset/{name}_train.csv'
            test_file = f'./data/eval_3_literature/dataset/{name}_test.csv'
            train_rdkit, test_rdkit = self.get_features(train_file, test_file, lit_name=name)
            model_dir = f'./trained_models/eval_3_literature/rf/{name}'
            trainer = RFTrainer(train_rdkit, test_rdkit, seed=self.seed1)
            best_params = trainer.model_tuning()
            params_dict[name] = best_params
            r2, mse, mae = trainer.testing_literature('rdkit', best_params, model_dir)
            performance_dict[name] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open('./trained_models/eval_3_literature/rf/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
        with open('./trained_models/eval_3_literature/rf/best_hyperparams.json', 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    def chemberta(self):
        performance_dict = {}
        for rep in range(self.n_repetition):
            seed = self.seed2 + rep
            for name in self.lit_names:
                train = pd.read_csv(f'./data/eval_3_literature/dataset/{name}_train.csv')
                test = pd.read_csv(f'./data/eval_3_literature/dataset/{name}_test.csv')
                save_dir = f'./trained_models/eval_3_literature/chemberta-2/rep_{rep}/{name}'
                trainer = CBERTaTrainer(train, test, model_name='DeepChem/ChemBERTa-77M-MTR')
                trainer.model_training(save_dir, train_batch_size=128, eval_batch_size=128, train_epoch=100, 
                                       eval_steps=10, max_length=195, seed=seed)
                r2, mse, mae = trainer.model_testing(save_dir, max_length=195)
                performance_dict[name] = (
                    f'R2: {r2:.4f} '
                    f'MAE: {mae:.4f} '
                    f'MSE: {mse:.4f}'
                )
            with open(f'./trained_models/eval_3_literature/chemberta-2/rep_{rep}/model_performance.json', 'w') as f:
                json.dump(performance_dict, f, indent=2)
    
    def graphormer(self):
        performance_dict = {}
        for name in self.lit_names:
            train_file = f'./data/eval_3_literature/dataset/{name}_train.csv'
            test_file = f'./data/eval_3_literature/dataset/{name}_test.csv'
            self.generate_graph(train_file, test_file, name)
            trainer = GraphormerTrainer(data_dir=f'./data/2D_graph_data/eval_3_literature/{name}', model_name='clefourrier/graphormer-base-pcqm4mv2')
            save_dir = f'./trained_models/eval_3_literature/graphormer/{name}'
            trainer.model_training(save_dir, train_batch_size=128, eval_batch_size=128, train_epoch=100, 
                                   eval_steps=10, seed=self.seed2)
            r2, mse, mae = trainer.model_testing(save_dir, test_file)
            performance_dict[name] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open('./trained_models/eval_3_literature/graphormer/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)

    # Seed for Uni-Mol needs to be changed internally as it does not take in seeds as arguments.
    # Please change it in the source code and run this method repeatedly with different seeds. Seeds used: 42, 43, 44
    def unimol(self, rep):
        performance_dict = {}
        for name in self.lit_names:
            train_file = f'./data/eval_3_literature/dataset/{name}_train.csv'
            test_file = f'./data/eval_3_literature/dataset/{name}_test.csv'
            trainer = UniMolTrainer(train_file, test_file)
            save_dir = f'./trained_models/eval_3_literature/unimol/rep_{rep}/{name}'
            trainer.model_training(save_dir, train_batch_size=16, train_epochs=100, early_stopping=10)
            r2, mse, mae = trainer.model_testing(save_dir)
            performance_dict[name] = (
                f'R2: {r2:.4f} '
                f'MAE: {mae:.4f} '
                f'MSE: {mse:.4f}'
            )
        with open(f'./trained_models/eval_3_literature/unimol/rep_{rep}/model_performance.json', 'w') as f:
            json.dump(performance_dict, f, indent=2)
    
    def init_all(self):
        self.lgbm()
        self.rf()
        self.chemberta()
        self.graphormer()
        self.unimol(rep=2)

def main():
    trainers = ModelTrainer()
    trainers.init_all()

if __name__ == "__main__":
    main()
