import numpy as np
import lightgbm as lgb
from pathlib import Path

if __name__ == '__main__':
	# Load data from numpy files
	data_dir = Path("./data")
	X = np.load(data_dir / "hand_features.npy")  # (1000, 17)
	X_cat = np.load(data_dir / "hand_categorical.npy").reshape(-1, 1)  # Reshape (1000,) to (1000, 1)
	y = np.load(data_dir / "hand_targets.npy")  # (1000,)

	# Combine features and define categorical column index
	X_combined = np.hstack([X_cat, X])
	categorical_indices = [0]  # X_cat is now the first column

	# Create LightGBM dataset
	train_data = lgb.Dataset(
			X_combined, 
			label=y,
			categorical_feature=categorical_indices
	)

	# Set parameters for regression
	params = {
			'objective': 'regression',
			'metric': 'rmse',
			'boosting_type': 'gbdt',
			'num_leaves': 31,
			'learning_rate': 0.05,
			'feature_fraction': 0.9,
			'min_data_in_leaf': 20,
			'verbose': -1
	}

	# Cross-validation
	cv_results = lgb.cv(
			params,
			train_data,
			num_boost_round=1000,
			nfold=5,
			stratified=False,
			shuffle=True,
			seed=42
	)

	best_rounds = len(cv_results['valid rmse-mean'])
	final_model = lgb.train(
			params,
			train_data,
			num_boost_round=best_rounds,
			callbacks=[lgb.log_evaluation(100)]
	)

	importance = final_model.feature_importance(importance_type='gain')
	feature_names = ['category'] + [f'num_{i}' for i in range(X.shape[1])]
	for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]:
			print(f"{name}: {imp}")

	final_model.save_model('lgb_hand_equity_model.txt')