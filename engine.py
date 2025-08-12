import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, GraphormerForGraphClassification
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from unimol_tools import MolTrain, MolPredict
from lightgbm import LGBMRegressor
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import pandas as pd
import optuna
import joblib
import gc
import os

class PredefinedFeatures:
    def __init__(self, data, train, test, save_dir):
        self.data = data
        self.train = train
        self.test = test
        self.save_dir = save_dir
        
    def rdkit_descs(self, train_file_name, test_file_name):
        train_file = os.path.join(self.save_dir, train_file_name)
        test_file = os.path.join(self.save_dir, test_file_name)
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            train_rdkit = pd.read_csv(train_file)
            test_rdkit = pd.read_csv(test_file)
        else:
            smi = self.data['SMILES']
            all_descs = [x[0] for x in Descriptors._descList]
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descs)
            desc_list = []
            for smile in smi:
                mol = Chem.MolFromSmiles(smile)
                desc = calc.CalcDescriptors(mol)
                desc_list.append(desc)
            df_desc = pd.DataFrame(desc_list, columns=all_descs)
            var_per_col = list(df_desc.var())
            zero_var_desc = []
            for idx, var in enumerate(var_per_col):
                if var == 0:
                    zero_var_desc.append(idx)
            df_desc = df_desc.drop(columns=df_desc.columns[zero_var_desc])
            df_desc['SMILES'] = self.data['SMILES']
            test_rdkit = pd.merge(self.test, df_desc, on='SMILES', how='left')
            train_rdkit = pd.merge(self.train, df_desc, on='SMILES', how='left')
            test_rdkit = test_rdkit.drop_duplicates(subset='SMILES')
            train_rdkit = train_rdkit.drop_duplicates(subset='SMILES') # Drop duplicates for literature datasets
            train_rdkit.to_csv(train_file, index=False)
            test_rdkit.to_csv(test_file, index=False)
        return train_rdkit, test_rdkit
    
    @staticmethod
    def rdkit_descs_transporter(ref_rdkit_train_file, transport_file, save_file):
        if os.path.isfile(save_file):
            transport_descs = pd.read_csv(save_file)
        else:
            ref_train_rdkit = pd.read_csv(ref_rdkit_train_file)
            transport = pd.read_csv(transport_file)
            smi = transport['SMILES']
            all_descs = [x[0] for x in Descriptors._descList]
            calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descs)
            desc_list = []
            for smile in smi:
                mol = Chem.MolFromSmiles(smile)
                desc = calc.CalcDescriptors(mol)
                desc_list.append(desc)
            transport_descs = pd.DataFrame(desc_list, columns=all_descs)
            transport_descs['SMILES'] = transport['SMILES']
            transport_descs['logPapp'] = transport['logPapp']
            ref_descs = ref_train_rdkit.columns
            transport_descs = transport_descs[ref_descs]
            transport_descs.to_csv(save_file, index=False)
        return transport_descs
        
    def morgan_cfps(self, train_file_name, test_file_name):
        train_file = os.path.join(self.save_dir, train_file_name)
        test_file = os.path.join(self.save_dir, test_file_name)
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            train_cfps = pd.read_csv(train_file)
            test_cfps = pd.read_csv(test_file)
        else:
            smi = self.data['SMILES']
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
            cfp_list = []
            for smile in smi:
                mol = Chem.MolFromSmiles(smile)
                cfp = mfpgen.GetCountFingerprint(mol)
                cfp = list(cfp)
                cfp_list.append(cfp)
            df_cfp = pd.DataFrame(cfp_list)
            df_cfp['SMILES'] = self.data['SMILES']
            test_cfps = pd.merge(self.test, df_cfp, on='SMILES', how='left')
            train_cfps = pd.merge(self.train, df_cfp, on='SMILES', how='left')
            train_cfps.to_csv(train_file, index=False)
            test_cfps.to_csv(test_file, index=False)
        return train_cfps, test_cfps

    def morgan_bitfps(self, train_file_name, test_file_name):
        train_file = os.path.join(self.save_dir, train_file_name)
        test_file = os.path.join(self.save_dir, test_file_name)
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            train_bitfps = pd.read_csv(train_file)
            test_bitfps = pd.read_csv(test_file)
        else:
            smi = self.data['SMILES']
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
            bitfp_list = []
            for smile in smi:
                mol = Chem.MolFromSmiles(smile)
                fps =  mfpgen.GetFingerprint(mol)
                fps = list(fps)
                bitfp_list.append(fps)
            df_bitfp = pd.DataFrame(bitfp_list)
            df_bitfp['SMILES'] = self.data['SMILES']
            test_bitfps = pd.merge(self.test, df_bitfp, on='SMILES', how='left')
            train_bitfps = pd.merge(self.train, df_bitfp, on='SMILES', how='left')
            train_bitfps.to_csv(train_file, index=False)
            test_bitfps.to_csv(test_file, index=False)
        return train_bitfps, test_bitfps

    def mordred_descs(self, train_file_name, test_file_name):
        train_file = os.path.join(self.save_dir, train_file_name)
        test_file = os.path.join(self.save_dir, test_file_name)
        if os.path.isfile(train_file) and os.path.isfile(test_file):
            train_mordred = pd.read_csv(train_file)
            test_mordred = pd.read_csv(test_file)
        else:
            return 'Please run conformer_engine & calc_mordred with the appropriate environment'
        return train_mordred, test_mordred





class GraphData:
    def __init__(self):
        self.possible_chirality = {'CHI_UNSPECIFIED': 0, 'CHI_TETRAHEDRAL_CW': 1, 
                                   'CHI_TETRAHEDRAL_CCW': 2, 'CHI_OTHER': 3, 'misc': 4}
        self.possible_hybridization = {'SP': 0, 'SP2': 1, 'SP3': 2, 'SP3D': 3, 'SP3D2': 4, 'misc': 5}
        self.possible_aromaticity = {'False': 0, 'True': 1}
        self.possible_ring = {'False': 0, 'True': 1}
        self.possible_bond_type = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3, 'misc':4}
        self.possible_bond_stereo = {'STEREONONE': 0, 'STEREOZ': 1, 'STEREOE': 2, 
                                     'STEREOCIS': 3, 'STEREOTRANS': 4, 'STEREOANY': 5}
        self.possible_conjugation = {'False': 0, 'True': 1}
    
    def get_bond_feats(self, bond):
        bond_type = self.possible_bond_type.get(str(bond.GetBondType()), self.possible_bond_type['misc'])
        bond_stereo = self.possible_bond_stereo[str(bond.GetStereo())]
        bond_conjugation = self.possible_conjugation[str(bond.GetIsConjugated())]
        return [bond_type, bond_stereo, bond_conjugation]
    
    def mol2graph(self, mol):
        mol_node_feat = []
        mol_bond_feat = []
        edge_idx = [[], []]
        for atom in mol.GetAtoms():
            #Node feat
            atomic_num = atom.GetAtomicNum()
            chirality = self.possible_chirality.get(str(atom.GetChiralTag()), self.possible_chirality['misc'])
            degree = atom.GetTotalDegree()
            formal_charge = atom.GetFormalCharge()
            num_H = atom.GetTotalNumHs()
            num_radical = atom.GetNumRadicalElectrons()
            hybridization = self.possible_hybridization.get(str(atom.GetHybridization()), self.possible_hybridization['misc'])
            aromaticity = self.possible_aromaticity[str(atom.GetIsAromatic())]
            is_ring = self.possible_ring[str(atom.IsInRing())]
            node_feat = [atomic_num, chirality, degree, formal_charge, num_H, num_radical, hybridization, aromaticity, is_ring]
            mol_node_feat.append(node_feat)
        for bond in mol.GetBonds():
            #Edge idx
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_idx[0] += [i, j]
            edge_idx[1] += [j, i]
            #Edge attr 
            edge_feat = self.get_bond_feats(bond)
            mol_bond_feat.append(edge_feat) #From i to j
            mol_bond_feat.append(edge_feat) #From j to i
        return edge_idx, mol_bond_feat, len(mol_node_feat), mol_node_feat
    
    def create_dataset(self, data):
        edge_idx = []
        edge_attr = []
        y = []
        num_nodes = []
        node_f = []
        for idx in tqdm(data.index):
            try:
                logpapp = [data['logPapp'][idx]]
                mol = Chem.MolFromSmiles(data['SMILES'][idx])
                ei, ea, nn, nf = self.mol2graph(mol)
            except:
                s = data['SMILES'][idx]
                print(f'Mol {s} cannot be converted into graph representation')
                continue
            edge_idx.append(ei)
            edge_attr.append(ea)
            y.append(logpapp)
            num_nodes.append(nn)
            node_f.append(nf)
        feat = Dataset.from_dict({'edge_index': edge_idx,
                                  'edge_attr': edge_attr,
                                  'y': y,
                                  'num_nodes': num_nodes,
                                  'node_feat': node_f})
        return feat
    
    def save_graph(self, train, test, graph_dir, splits=5):
        test_dict = self.create_dataset(test)
        kfold = KFold(n_splits=splits)
        for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(train)):
            train_df = train.iloc[train_idx,:]
            valid_df = train.iloc[valid_idx,:]
            train_dict = self.create_dataset(train_df)
            valid_dict = self.create_dataset(valid_df)
            dataset_dict = DatasetDict({'train': train_dict,
                                        'validation': valid_dict,
                                        'test': test_dict})
            dataset_dict.save_to_disk(os.path.join(graph_dir, f'fold_{fold_no}'))
            
    
    
    
    
class LGBMTrainer:
    def __init__(self, train, test, seed):
        self.train = train
        self.test = test
        self.seed = seed
        
    def model_tuning(self):
        def objective(trial):
            params = {'num_leaves': trial.suggest_int('num_leaves', 10, 65),
                      'min_child_samples': trial.suggest_int('min_child_samples', 10, 65),
                      'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                      'learning_rate':trial.suggest_float('learning_rate', 0.01, 0.1)
                      }
            x = self.train.drop(columns=['SMILES', 'logPapp'])
            y = self.train['logPapp']
            kfold = KFold(n_splits=5)
            lgb_model = LGBMRegressor(num_leaves=params['num_leaves'], 
                                      min_child_samples=params['min_child_samples'], 
                                      n_estimators=params['n_estimators'],
                                      learning_rate=params['learning_rate'],
                                      feature_fraction=0.8,
                                      seed=self.seed)
            total_mse = 0
            for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(x)):
                x_train, x_valid = x.iloc[train_idx,:], x.iloc[valid_idx,:]
                y_train, y_valid = y[train_idx], y[valid_idx]
                lgb_model.fit(x_train, y_train)
                y_pred = lgb_model.predict(x_valid)
                mse = mean_squared_error(y_valid, y_pred)
                total_mse += mse
            return mse/5
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        trial_ = study.best_trial
        return trial_.params
    
    def model_testing(self, desc_name, best_params, model_dir):
        x_train = self.train.drop(columns=['SMILES', 'logPapp'])
        y_train = self.train['logPapp']
        x_test = self.test.drop(columns=['SMILES', 'logPapp'])
        y_test = self.test['logPapp']
        kfold = KFold(n_splits=5)
        lgb_model = LGBMRegressor(num_leaves=best_params['num_leaves'], 
                                  min_child_samples=best_params['min_child_samples'], 
                                  n_estimators=best_params['n_estimators'],
                                  learning_rate=best_params['learning_rate'],
                                  feature_fraction=0.8,
                                  seed=self.seed)
        r2 = 0
        mae = 0
        mse = 0
        for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(x_train)):
            x_train_fold, x_valid_fold = x_train.iloc[train_idx,:], x_train.iloc[valid_idx,:]
            y_train_fold, y_valid_fold = y_train[train_idx], y_train[valid_idx]
            lgb_model.fit(x_train_fold, y_train_fold)
            joblib.dump(lgb_model, os.path.join(model_dir, f'{desc_name}_trained_model_fold_{fold_no}.pkl'))
            y_pred = lgb_model.predict(x_test)
            r2 += r2_score(y_test, y_pred)
            mae += mean_absolute_error(y_test, y_pred)
            mse += mean_squared_error(y_test, y_pred)
        return r2/5, mse/5, mae/5
    
    def testing_literature(self, desc_name, best_params, model_dir): # Literature uses full data (no fold split)
        x_train = self.train.drop(columns=['SMILES', 'logPapp'])
        y_train = self.train['logPapp']
        x_test = self.test.drop(columns=['SMILES', 'logPapp'])
        y_test = self.test['logPapp']
        lgb_model = LGBMRegressor(num_leaves=best_params['num_leaves'], 
                                  min_child_samples=best_params['min_child_samples'], 
                                  n_estimators=best_params['n_estimators'],
                                  learning_rate=best_params['learning_rate'])
        lgb_model.fit(x_train, y_train)
        joblib.dump(lgb_model, os.path.join(model_dir, f'{desc_name}_trained_model.pkl'))
        y_pred = lgb_model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return r2, mse, mae
    
    def top_desc_training(self, best_params, trained_model_file):
        x_train = self.train.drop(columns=['SMILES', 'logPapp'])
        y_train = self.train['logPapp']
        x_test = self.test.drop(columns=['SMILES', 'logPapp'])
        y_test = self.test['logPapp']
        lgb_model = LGBMRegressor(num_leaves=best_params['num_leaves'], 
                                  min_child_samples=best_params['min_child_samples'], 
                                  n_estimators=best_params['n_estimators'],
                                  learning_rate=best_params['learning_rate'],
                                  feature_fraction=0.8,
                                  seed=self.seed)
        lgb_model.fit(x_train, y_train)
        y_pred = lgb_model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return r2, mse, mae
    
    def model_predict(self, model_file):
        pred_data = self.test.drop(columns=['SMILES', 'logPapp'])
        model = joblib.load(model_file)
        y_pred = model.predict(pred_data)
        return y_pred





class RFTrainer:
    def __init__(self, train, test, seed):
        self.train = train
        self.test = test
        self.seed = seed
    
    def model_tuning(self):
        def objective(trial):
            params = {'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                      'max_depth': trial.suggest_int('max_depth', 10, 200)
                      }
            x = self.train.drop(columns=['SMILES', 'logPapp'])
            y = self.train['logPapp']
            kfold = KFold(n_splits=5)
            rf_model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                              max_depth=params['max_depth'],
                                              random_state=self.seed,
                                              n_jobs=-1)
            total_mse = 0
            for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(x)):
                x_train, x_valid = x.iloc[train_idx,:], x.iloc[valid_idx,:]
                y_train, y_valid = y[train_idx], y[valid_idx]
                rf_model.fit(x_train, y_train)
                y_pred = rf_model.predict(x_valid)
                mse = mean_squared_error(y_valid, y_pred)
                total_mse += mse
            return mse/5
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=60)
        trial_ = study.best_trial
        return trial_.params
    
    def model_testing(self, desc_name, best_params, model_dir):
        x_train = self.train.drop(columns=['SMILES', 'logPapp'])
        y_train = self.train['logPapp']
        x_test = self.test.drop(columns=['SMILES', 'logPapp'])
        y_test = self.test['logPapp']
        kfold = KFold(n_splits=5)
        rf_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                          max_depth=best_params['max_depth'],
                                          random_state=self.seed,
                                          n_jobs=-1)
        r2 = 0
        mae = 0
        mse = 0
        for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(x_train)):
            x_train_fold, x_valid_fold = x_train.iloc[train_idx,:], x_train.iloc[valid_idx,:]
            y_train_fold, y_valid_fold = y_train[train_idx], y_train[valid_idx]
            rf_model.fit(x_train_fold, y_train_fold)
            joblib.dump(rf_model, os.path.join(model_dir, f'{desc_name}_trained_model_fold_{fold_no}.pkl'))
            y_pred = rf_model.predict(x_test)
            r2 += r2_score(y_test, y_pred)
            mae += mean_absolute_error(y_test, y_pred)
            mse += mean_squared_error(y_test, y_pred)
        return r2/5, mse/5, mae/5
    
    def testing_literature(self, desc_name, best_params, model_dir):
        x_train = self.train.drop(columns=['SMILES', 'logPapp'])
        y_train = self.train['logPapp']
        x_test = self.test.drop(columns=['SMILES', 'logPapp'])
        y_test = self.test['logPapp']
        rf_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                          max_depth=best_params['max_depth'],
                                          random_state=self.seed,
                                          n_jobs=-1)
        rf_model.fit(x_train, y_train)
        joblib.dump(rf_model, os.path.join(model_dir, f'{desc_name}_trained_model.pkl'))
        y_pred = rf_model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return r2, mse, mae
    
    def top_desc_training(self, best_params, trained_model_file):
        x_train = self.train.drop(columns=['SMILES', 'logPapp'])
        y_train = self.train['logPapp']
        x_test = self.test.drop(columns=['SMILES', 'logPapp'])
        y_test = self.test['logPapp']
        rf_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                          max_depth=best_params['max_depth'],
                                          random_state=self.seed,
                                          n_jobs=-1)
        rf_model.fit(x_train, y_train)
        y_pred = rf_model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return r2, mse, mae
    
    def model_predict(self, model_file):
        pred_data = self.test.drop(columns=['SMILES', 'logPapp'])
        model = joblib.load(model_file)
        y_pred = model.predict(pred_data)
        return y_pred





class CBERTaInput(TorchDataset):
    def __init__(self, i_data, i_tokenizer, i_max_length):
        self.data = i_data
        self.tokenizer = i_tokenizer
        self.max_length = i_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['SMILES']
        inputs = self.tokenizer(smiles, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = inputs['token_type_ids'].squeeze(0)
        inputs['labels'] = torch.tensor(self.data.iloc[idx]['logPapp'], dtype=torch.float).unsqueeze(0)
        return inputs
    
    
    
    
    
class CBERTaTrainer:
    def __init__(self, train, test, model_name):
        self.train = train
        self.test = test
        self.model_name = model_name
    
    def model_training(self, save_dir, train_batch_size, eval_batch_size, train_epoch, eval_steps, max_length, seed, folds=5):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_hidden_layers += 1
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
        model.to('cuda')
        kf = KFold(n_splits=folds)
        for fold_no, (train_idx, valid_idx) in enumerate(kf.split(self.train)):
            train_data = self.train.iloc[train_idx,:]
            valid_data = self.train.iloc[valid_idx,:]
            train_dataset = CBERTaInput(train_data, tokenizer, max_length)
            validation_dataset = CBERTaInput(valid_data, tokenizer, max_length)
            training_args = TrainingArguments(
                output_dir=os.path.join(save_dir, f'fold_{fold_no}'),
                optim='adamw_torch',
                num_train_epochs=train_epoch,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                logging_steps=10,
                eval_steps=eval_steps,
                save_steps=1000,
                seed=seed,
                evaluation_strategy='steps',
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
            )
            trainer.train()
            trainer.save_model(os.path.join(save_dir, f'fold_{fold_no}'))
            gc.collect()
            torch.cuda.empty_cache()
    
    def model_testing(self, trained_dir, max_length, folds=5):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_hidden_layers += 1
        r2_results = []
        mae_results = []
        mse_results = []
        for fold_no in range(folds):
            model_loc = os.path.join(trained_dir, f'fold_{fold_no}')
            model = AutoModelForSequenceClassification.from_pretrained(model_loc)
            model.to('cuda')
            test_smiles = self.test['SMILES']
            predictions = []
            for smiles in test_smiles:
                inputs = tokenizer(smiles, return_tensors='pt', padding='max_length', truncation=True, max_length=195).to('cuda') #max_length=195
                with torch.no_grad():
                    outputs = model(**inputs)
                predicted_property = outputs.logits.squeeze().item()
                predictions.append(predicted_property)
            predictions_df = pd.DataFrame(predictions, columns=['pred_logPapp'])
            predictions_df['logPapp'] = self.test['logPapp']
            predictions_df['SMILES'] = self.test['SMILES']
            predictions_df.to_csv(os.path.join(model_loc, 'predictions.csv'), index=False)
            mse_results.append(mean_squared_error(self.test['logPapp'], predictions))
            mae_results.append(mean_absolute_error(self.test['logPapp'], predictions))
            r2_results.append(r2_score(self.test['logPapp'], predictions))
            gc.collect()
            torch.cuda.empty_cache()
        return np.mean(r2_results), np.mean(mse_results), np.mean(mae_results)





class GraphormerTrainer:
    def __init__(self, data_dir, model_name):
        self.data_dir = data_dir
        self.model_name = model_name
        
    def model_training(self, save_dir, train_batch_size, eval_batch_size, train_epoch, eval_steps, seed, folds=5):
        model = GraphormerForGraphClassification.from_pretrained(
            self.model_name,
            num_classes=1,
            ignore_mismatched_sizes=True,
        )
        model.to('cuda')
        for fold_no in range(folds):
            data = load_from_disk(os.path.join(self.data_dir, f'fold_{fold_no}'))
            data_processed = data.map(preprocess_item, batched=False)
            training_args = TrainingArguments(
                output_dir=os.path.join(save_dir, f'fold_{fold_no}'),
                optim="adamw_torch",
                num_train_epochs=train_epoch,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=eval_batch_size,
                logging_steps=10,
                eval_steps=eval_steps,
                save_steps=1000,
                seed=seed,
                evaluation_strategy="steps",
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=data_processed["train"],
                eval_dataset=data_processed["validation"],
                data_collator=GraphormerDataCollator(),
            )
            trainer.train()
            trainer.save_model(os.path.join(save_dir, f'fold_{fold_no}'))
            gc.collect()
            torch.cuda.empty_cache()
    
    def model_testing(self, trained_dir, test_file, batch_size=64, folds=5):
        r2_results = []
        mae_results = []
        mse_results = []
        test = pd.read_csv(test_file)
        data = load_from_disk(os.path.join(self.data_dir, 'fold_0')) # Test data is the same in each fold split
        data_processed = data.map(preprocess_item, batched=False)
        test_batch = DataLoader(data_processed['test'], batch_size=batch_size, collate_fn=GraphormerDataCollator())
        for fold_no in range(folds):
            model_loc = os.path.join(trained_dir, f'fold_{fold_no}')
            model = GraphormerForGraphClassification.from_pretrained(model_loc)
            model.to('cuda')
            pred_list = []
            for data in test_batch:
                batch = {k: v.to('cuda') for k, v in data.items()}
                with torch.no_grad():
                    out = model(input_nodes=batch['input_nodes'], input_edges=batch['input_edges'], attn_bias=batch['attn_bias'], 
                                in_degree=batch['in_degree'], out_degree=batch['out_degree'], spatial_pos=batch['spatial_pos'], 
                                attn_edge_type=batch['attn_edge_type'])
                pred = out[0].cpu().flatten().tolist()
                pred_list += pred
            predictions_df = pd.DataFrame(pred_list, columns=['pred_logPapp'])
            true_list = torch.tensor(data_processed['test']['y']).flatten().tolist()
            predictions_df['logPapp'] = true_list
            predictions_df['SMILES'] = test['SMILES']
            predictions_df.to_csv(os.path.join(model_loc, 'predictions.csv'), index=False)
            r2_results.append(r2_score(true_list, pred_list))
            mae_results.append(mean_absolute_error(true_list, pred_list))
            mse_results.append(mean_squared_error(true_list, pred_list))
            gc.collect()
            torch.cuda.empty_cache()
        return np.mean(r2_results), np.mean(mse_results), np.mean(mae_results)
        

        
        
        
class UniMolTrainer:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
    
    def model_training(self, save_dir, train_batch_size, train_epochs, early_stopping, folds=5):
        model = MolTrain(task='regression', epochs=train_epochs, learning_rate=0.0001, batch_size=train_batch_size, early_stopping=early_stopping,
                       metrics='mse', kfold=folds, save_path=save_dir, smiles_col='SMILES',
                       model_name='unimolv1', model_size='84m', target_col_prefix='logPapp')
        model.fit(self.train_file)
    
    def model_testing(self, trained_dir):
        model = MolPredict(load_model=trained_dir)
        res = model.predict(self.test_file)
        test = pd.read_csv(self.test_file)
        test_data = test[['logPapp', 'SMILES']]
        test_data['pred_logPapp'] = res
        test_data.to_csv(os.path.join(trained_dir, 'predictions.csv'), index=False)
        r2 = r2_score(test_data['logPapp'], test_data['pred_logPapp'])
        mae = mean_absolute_error(test_data['logPapp'], test_data['pred_logPapp'])
        mse = mean_squared_error(test_data['logPapp'], test_data['pred_logPapp'])
        return r2, mse, mae

