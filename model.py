import mlflow
import mlflow.sklearn
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import optuna

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

# Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Preprocessing
X = X / 255.0  # Normalize pixel values
y = y.astype(int)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Define the objective function for Optuna
def objective(trial):
    with mlflow.start_run(nested=True) as child_run:
        # Suggest hyperparameters
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)])
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
        learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-4, 1e-2)

        # Initialize the model
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=20,
            random_state=42
        )
        
        # Evaluate using cross-validation on the validation set
        validation_accuracy = cross_val_score(model, X_val, y_val, cv=3, n_jobs=-1).mean()

        # Log parameters and metrics to MLflow
        mlflow.log_params(trial.params)
        mlflow.log_metric('validation_accuracy', validation_accuracy)

        return validation_accuracy

# Set up MLflow experiment
experiment_name = "Optuna_MNIST_Hyperparameter_Tuning"
client = MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)
if experiment:
    if experiment.lifecycle_stage == 'deleted':
        print(f"Restoring deleted experiment: {experiment_name}")
        client.restore_experiment(experiment.experiment_id)
    print(f"Using existing experiment: {experiment_name}")
else:
    print(f"Creating new experiment: {experiment_name}")

mlflow.set_experiment(experiment_name)

# Start a parent run
with mlflow.start_run(run_name="Optuna_MNIST_Hyperparameter_Tuning") as parent_run:
    # Create an Optuna study
    study = optuna.create_study(direction='maximize')
    
    # Optimize the objective function with 10 trials
    study.optimize(objective, n_trials=10)

    # Fetch experiment and parent run ID dynamically
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    parent_run_id = mlflow.active_run().info.run_id

    # Fetch all child runs using the correct parent run ID
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{parent_run_id}'"
    )

    # Prepare data for visualization
    runs['validation_accuracy'] = runs['metrics.validation_accuracy'].astype(float)

    # Identify the best hyperparameter set
    best_run_id = runs.loc[runs['validation_accuracy'].idxmax(), 'run_id']
    best_accuracy = runs['validation_accuracy'].max()

    plt.figure(figsize=(10, 6))

    # Plot all hyperparameter sets with validation accuracy
    plt.plot(runs['run_id'], runs['validation_accuracy'], marker='o', linestyle='-', color='blue', label='Validation Accuracy')

    # Highlight the best validation score with gold color
    best_index = runs.index[runs['run_id'] == best_run_id][0]
    plt.plot(best_index, best_accuracy, 'o', color='gold', markersize=12, label='Best Validation Score')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Child Run ID')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Child Run ID (Best in Gold)')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    # Save the plot as a PNG image in the current directory
    plot_path = "validation_accuracy_vs_runs.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path, "plots")
    print(f"Plot saved as {plot_path}")

    plt.close()
