## zenml

1.
`zenml init`

2.
`zenml project register class_llm --set`

3.
`zenml login --local --docker `  
or:   
`docker run -it -d -p 8080:8080 --name zenml zenmldocker/zenml-server `   
`zenml login http://localhost:8080`

### mlflow integration

1.
`zenml integration install mlflow -y`

2.
`zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow`

3. dashboard öffnen
`mlflow server`

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
   - `zenml metadata-store list`
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