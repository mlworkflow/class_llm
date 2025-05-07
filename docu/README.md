## zenml

1.
`zenml init`
or
`zenml login http://172.201.218.136:8080 `
2.
`zenml project register class_llm --set`

3.
`zenml login --local --docker `  
or:   
`docker run -it -d -p 8080:8080 --name zenml zenmldocker/zenml-server `   
`zenml login http://localhost:8080`
`zenml login --refresh http://localhost:8080`

### mlflow integration

1.
`zenml integration install mlflow -y`

2.
`zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow`
`zenml experiment-tracker register mlflow_experiment_tracker2 --flavor=mlflow --tracking_uri=http://172.201.218.136:5000 --tracking_token=dummy`

1. dashboard öffnen
`mlflow server`

1. mit remote server verbinden
`mlflow.set_tracking_uri('https://dbc-678b3979-34b1.cloud.databricks.com/browse/folders/workspace?o=300400946232800')`  
`mlflow.set_experiment("https://dbc-678b3979-34b1.cloud.databricks.com/ml/experiments/3034448658823987")`

### stack erstellen

To successfully register a stack in ZenML, you need to specify all required stack components in addition to the `experiment_tracker`. These components include:

1. **Orchestrator** (`-o`): Responsible for running the pipeline (e.g., `local`, `airflow`, `kubeflow`).
2. **Artifact Store** (`-a`): Used to store pipeline artifacts (e.g., local filesystem or cloud storage like S3, GCS).
3. **Metadata Store** (`-m`): Tracks metadata and pipeline lineage (e.g., `sqlite`, `mysql`, `postgresql`).
4. **Optional Components**:
   - **Secrets Manager** (`-s`): Manages sensitive credentials (e.g., `local`, `aws_secrets_manager`).
   - **Container Registry** (`-c`): Stores Docker images for pipelines (e.g., Docker Hub, AWS ECR).
   - **Step Operator** (`-so`): Executes steps in custom environments (e.g., cloud-based runtimes).

---
Here’s how you can register and set a stack with all essential components:

```bash
zenml stack register stack1 --orchestrator=default --artifact-store=default -e mlflow_experiment_tracker --set
```

---

#### Explanation of Parameters
1. **Orchestrator (`--orchestrator` or `-o`)**:
   - Example: `local_orchestrator` for running pipelines locally.
   - Use `zenml orchestrator list` to see available orchestrators.

2. **Artifact Store (`--artifact-store` or `-a`)**:
   - Example: `local_artifact_store` for storing artifacts locally.
   - Use `zenml artifact-store list` to see available artifact stores.

3. **Metadata Store (`--metadata-store` or `-m`)**:
   - Example: `sqlite_metadata_store` for lightweight, local metadata storage.

4. **Experiment Tracker (`--experiment-tracker` or `-e`)**:
   - Example: `mlflow_experiment_tracker` for MLflow-based experiment tracking.

5. **Optional Components**:
   - **Secrets Manager (`--secrets-manager` or `-s`)**:
     - Example: `local_secrets_manager` for managing credentials locally.
   - **Container Registry (`--container-registry` or `-c`)**:
     - Example: `docker_hub_registry` for storing container images.
   - **Step Operator (`--step-operator` or `-so`)**:
     - Example: `kubernetes_step_operator` for running steps on Kubernetes.

---

#### Steps to Identify Available Components
1. **List Available Components**:
   Run the following commands to see the available components:
   - `zenml orchestrator list`
   - `zenml artifact-store list`
   - `zenml experiment-tracker list`
   - `zenml secrets-manager list`
   - `zenml container-registry list`
   - `zenml step-operator list`

2. **Register Missing Components**:
   If required components are not available, register them first. For example:
   ```bash
   zenml orchestrator register local_orchestrator --type=local
   zenml artifact-store register local_artifact_store --type=local --path=/path/to/store
   zenml metadata-store register sqlite_metadata_store --type=sqlite --path=/path/to/sqlite.db
   ```

### run in colab

#### Approach: Split Pipeline with Artifact Store

The best approach is to:
1. Create a "local pipeline" that runs all steps except training
2. Use a shared artifact store that both environments can access
3. Create a "Colab pipeline" with an importer step that accesses artifacts from the shared store

Step 1: Setup Shared Artifact Store

First, you need to configure a stack with an artifact store that both your local environment and Colab can access. Google Drive is a good choice since it's easily accessible from Colab.

```bash
# Register a Google Drive artifact store for local use
zenml artifact-store register gdrive_artifact_store --flavor=local --path="G:\Meine Ablage\zenml\artifacts"



# Register a stack that uses this artifact store
zenml stack register preproc4colab \
    --artifact-store=gdrive_artifact_store \
    --orchestrator=default \
    -e mlflow_experiment_tracker \
    --set

zenml stack register stack2 --artifact-store=gdrive_artifact_store --orchestrator=default -e mlflow_experiment_tracker2 --set
```

Step 2: Create Local Pipeline

Step 3: Create Importer Step for Colab

Create a new importer step file `steps/_importer.py`:

Step 4: Create Colab Pipeline

Step 5: Create Colab Notebook

Create a Colab notebook file `train_in_colab.ipynb`:

#### How to Use This Setup

1. **Run locally first**:
   ```bash
   python local_pipeline.py
   ```
   This will run all preprocessing steps and store the artifacts in your Google Drive. Note the run ID that is printed at the end.

2. **Open the Colab notebook**:
   - Upload `train_in_colab.ipynb` to Google Colab
   - Update the `LOCAL_PIPELINE_RUN_ID` with the ID from step 1
   - Run all cells in the notebook

This approach gives you several advantages:
- You only need to do the data preprocessing once locally
- The training step in Colab has access to all necessary artifacts
- You maintain the ZenML pipeline structure and tracking
- You can easily switch between environments

Alternative approaches could include:
1. Using a cloud artifact store (like S3 or GCS) instead of Google Drive
2. Using a remote metadata store that both environments can access
3. Splitting the pipeline into two separate pipelines with different triggers
