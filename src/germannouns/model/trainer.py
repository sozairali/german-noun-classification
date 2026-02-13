"""
Model training for German noun gender classification.

Trains decision tree and random forest models with cost complexity pruning,
grid search hyperparameter tuning, and ensemble methods.
"""

from typing import Tuple, Dict, Any
import pickle
import multiprocessing
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from germannouns.config import (
    TRAIN_TEST_SPLIT,
    VALIDATION_SPLIT,
    RANDOM_STATE,
    CCP_ALPHA,
    GRID_SEARCH_PARAMS,
    RF_GRID_SEARCH_PARAMS,
    RF_RANDOM_SEARCH_ITERATIONS,
    RF_RANDOM_STATE,
    RF_CV_FOLDS,
    N_JOBS,
    MODEL_PICKLE,
    DEFAULT_PLOT_SIZE
)
from germannouns.data.loader import load_nouns, get_gender_distribution
from germannouns.data.feature_engineering import prepare_training_data


def train_decision_tree(ccp_alpha: float, X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Train a single decision tree with given CCP alpha.

    Args:
        ccp_alpha: Cost complexity pruning parameter
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained DecisionTreeClassifier
    """
    dt_classifier = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=RANDOM_STATE)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier


def choose_ccp_alpha(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Visualize accuracy vs alpha to choose optimal CCP alpha value.

    Creates a plot showing train and test accuracy for different alpha values
    from cost complexity pruning path. User selects alpha by eye-balling the plot.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        None (displays plot)
    """
    n_cores = multiprocessing.cpu_count()

    # Fit initial tree to get pruning path
    tre = DecisionTreeClassifier()
    tre.fit(X_train, y_train)

    # Get cost complexity pruning path
    path = tre.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = [max(0, x) for x in ccp_alphas]

    # Train trees for each alpha value
    print(f"Training {len(ccp_alphas)} trees with different alpha values...")
    tres = []
    for count, ccp_alpha in enumerate(ccp_alphas):
        if count % 100 == 0:
            print(f"Progress: {count}/{len(ccp_alphas)}")
        clf = tree.DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        tres.append(clf)

    # Remove last element (trivial tree)
    tres = tres[:-1]
    ccp_alphas = ccp_alphas[:-1]

    # Calculate train and test accuracy for each alpha
    train_acc = []
    test_acc = []
    for c in tres:
        y_train_pred = c.predict(X_train)
        y_test_pred = c.predict(X_test)
        train_acc.append(accuracy_score(y_train_pred, y_train))
        test_acc.append(accuracy_score(y_test_pred, y_test))

    # Plot accuracy vs alpha
    plt.scatter(ccp_alphas, train_acc)
    plt.scatter(ccp_alphas, test_acc)
    plt.plot(ccp_alphas, train_acc, label='train_accuracy', drawstyle="steps-post")
    plt.plot(ccp_alphas, test_acc, label='test_accuracy', drawstyle="steps-post")
    plt.legend()
    plt.title('Accuracy vs alpha')
    plt.xlabel('CCP Alpha')
    plt.ylabel('Accuracy')
    plt.show()


def visualize(model: DecisionTreeClassifier, features: pd.DataFrame,
             categories: pd.Series, size: int = DEFAULT_PLOT_SIZE) -> None:
    """
    Visualize decision tree and print feature importance statistics.

    Args:
        model: Trained DecisionTreeClassifier
        features: Feature DataFrame (for column names)
        categories: Category labels (for class names)
        size: Figure size for plot

    Returns:
        None (displays plot and prints statistics)
    """
    # Visualize the tree
    plt.figure(figsize=(size, size))
    plot_tree(model, filled=True, feature_names=features.columns,
             class_names=categories.unique())
    plt.show()

    # Analyze feature importance
    print("Feature Importance:")
    for feature, importance in zip(features.columns, model.feature_importances_):
        if importance > 0:  # Only show features actually used
            print(f"{feature}: {importance:.4f}")

    # Analyze node distribution
    print("\nNode Distribution:")
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of leaf nodes: {sum(children_left == -1)}")

    # Analyze leaf node statistics
    leaf_samples = model.tree_.n_node_samples[children_left == -1]
    class_distribution = model.tree_.value[children_left == -1]
    print("\nLeaf Node Statistics:")
    print(f"Average samples per leaf: {leaf_samples.mean():.2f}")
    print(f"Min samples in leaf: {leaf_samples.min()}")
    print(f"Max samples in leaf: {leaf_samples.max()}")


def run_GSCV_DecisionTree(params: Dict[str, Any], X_train: pd.DataFrame,
                         y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Run Grid Search Cross Validation for Decision Tree hyperparameters.

    Args:
        params: Dictionary of hyperparameters to search
        X_train: Training features
        y_train: Training labels

    Returns:
        Best DecisionTreeClassifier from grid search
    """
    n_cores = multiprocessing.cpu_count()
    tre_gcv = tree.DecisionTreeClassifier()
    gcv = GridSearchCV(estimator=tre_gcv, param_grid=params, n_jobs=n_cores)
    gcv.fit(X_train, y_train)

    print(f"Best parameters: {gcv.best_params_}")
    return gcv.best_estimator_


def save_model(model: DecisionTreeClassifier, features: list,
              model_path: str = MODEL_PICKLE) -> None:
    """
    Save trained model and feature list to pickle file.

    Args:
        model: Trained model
        features: List of feature names
        model_path: Path to save pickle file

    Returns:
        None
    """
    with open(model_path, 'wb') as file:
        pickle.dump((model, features), file)
    print(f"Model saved to: {model_path}")


def train_model(X: pd.DataFrame, y: pd.Series, features: list) -> DecisionTreeClassifier:
    """
    Complete model training pipeline.

    Trains multiple models:
    1. Decision tree with cost complexity pruning
    2. Decision tree with grid search (pre-pruning)
    3. Random forest with random search
    4. Ensemble of decision tree + random forest

    Args:
        X: Feature matrix
        y: Gender labels
        features: List of feature names

    Returns:
        Best decision tree model (with CCP pruning)
    """
    # Split data into train, test, and validation sets (3:1:1 split)
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Val size: {len(X_val)}")

    # 1. COST COMPLEXITY PRUNING (POST-PRUNING)
    print("\n=== Cost Complexity Pruning ===")
    print("Visualizing alpha selection (close plot to continue)...")
    choose_ccp_alpha(X_train, y_train, X_test, y_test)

    # Train with chosen alpha
    print(f"\nTraining decision tree with CCP alpha={CCP_ALPHA}...")
    tre_ccp = tree.DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=CCP_ALPHA)
    tre_ccp.fit(X_train, y_train)

    y_train_pred = tre_ccp.predict(X_train)
    y_test_pred = tre_ccp.predict(X_test)

    print(f'Post-pruning train accuracy: {accuracy_score(y_train_pred, y_train):.4f}')
    print(f'Post-pruning test accuracy: {accuracy_score(y_test_pred, y_test):.4f}')
    print(f"Tree depth: {tre_ccp.get_depth()}")

    # Save model
    save_model(tre_ccp, features)

    # Visualize tree
    print("\nVisualizing tree (close plot to continue)...")
    visualize(tre_ccp, X, y, DEFAULT_PLOT_SIZE)

    # 2. GRID SEARCH (PRE-PRUNING)
    print("\n=== Grid Search CV ===")
    print("Running grid search...")
    tre_gcv = run_GSCV_DecisionTree(GRID_SEARCH_PARAMS, X_train, y_train)

    y_train_pred = tre_gcv.predict(X_train)
    y_test_pred = tre_gcv.predict(X_test)

    print(f'GCV train accuracy: {accuracy_score(y_train_pred, y_train):.4f}')
    print(f'GCV test accuracy: {accuracy_score(y_test_pred, y_test):.4f}')

    # 3. RANDOM FOREST
    print("\n=== Random Forest ===")
    print("Running random search...")
    n_cores = multiprocessing.cpu_count()

    rf = RandomForestClassifier()
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=RF_GRID_SEARCH_PARAMS,
        n_iter=RF_RANDOM_SEARCH_ITERATIONS,
        cv=RF_CV_FOLDS,
        verbose=2,
        random_state=RF_RANDOM_STATE,
        n_jobs=n_cores
    )

    random_search.fit(X_train, y_train)
    print(f"Best RF parameters: {random_search.best_params_}")

    rf_best = random_search.best_estimator_
    y_test_pred_rf = rf_best.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_test_pred_rf)
    print(f"Random forest test accuracy: {rf_accuracy:.4f}")

    # 4. ENSEMBLE MODEL
    print("\n=== Ensemble Model ===")
    ensemble = VotingClassifier(
        estimators=[('decision_tree', tre_ccp), ('random_forest', rf_best)],
        voting='hard'
    )
    ensemble.fit(X_train, y_train)
    ensemble_preds = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    print(f"Ensemble test accuracy: {ensemble_accuracy:.4f}")

    print("\n=== Training Complete ===")
    print(f"Best model saved to: {MODEL_PICKLE}")

    return tre_ccp


def main():
    """
    Main entry point for model training script.

    Loads data, prepares features, and trains models.
    """
    print("Loading and cleaning nouns dataset...")
    nouns = load_nouns()
    print(f"Loaded {len(nouns)} nouns")

    # Get gender distribution for chi-square test
    gender_dist = get_gender_distribution(nouns)
    print("\nGender distribution:")
    print(gender_dist)

    # Prepare training data (feature engineering)
    X, y, features = prepare_training_data(nouns, gender_dist)

    # Train models
    model = train_model(X, y, features)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
