Deploying a ZenML server on an Azure Virtual Machine involves several steps, from setting up the VM itself to configuring ZenML. Here's a general guide to help you through the process.

1.  **Connect to your Azure VM:**
    *   Once the VM is running, connect to it using SSH:
        ```bash
        ssh your_username@your_vm_public_ip
        ```


2.  **Install Docker on the VM:**
    If Docker isn't already installed on your VM:
    ```bash
    # Update package lists
    sudo apt update

    # Install Docker
    sudo apt install docker.io -y
    sudo systemctl start docker
    sudo systemctl enable docker

    # Add your user to the docker group to run docker commands without sudo (optional, requires logout/login)
    sudo usermod -aG docker $USER
    # You might need to log out and log back in for this change to take effect,
    # or you can just continue using 'sudo docker ...' for the commands below.
    ```

3.  **Configure Network Security Group (NSG):**
    *   In the Azure portal, go to your VM's settings.
    *   Find the "Networking" settings and your Network Security Group.
    *   Add an **inbound port rule** to allow traffic on **TCP port 8080** (this is the port ZenML server will use as per your Dockerfile).
        *   Source: Any (or a specific IP range if you want to restrict access)
        *   Source port ranges: \*
        *   Destination: Any (or your VM's private IP)
        *   Destination port ranges: 8080
        *   Protocol: TCP
        *   Action: Allow
        *   Priority: Choose a suitable number (e.g., 100).
        *   Name: A descriptive name like `Allow-ZenML-8080`.

4.  **Install Git on the VM:**
    ```bash
    sudo apt update
    sudo apt install -y git
    ```

**Phase 2: Build and Run ZenML Server Docker Image on the VM**

1.  **Connect to your VM via SSH.**

2.  **Clone your Git Repository (optional):**
    ```bash
   git clone https://github.com/zenml-io/zenml.git
    cd zenml
    ```

3.  **Build the Docker Image (optional):**
`sudo docker build -t zenml-server-official -f docker/zenml-server-dev.Dockerfile --target runtime .`

4.  **Run the Docker Container from the Built Image:**
    ```bash
    sudo docker run -d -p 8080:8080 \
        --name my-zenml-server \
        -v zenml_server_data:/zenml/.zenconfig \
        zenml-server-official
    ```
    *   `sudo docker run`: Command to run a new container.
    *   `-d`: Runs the container in detached mode (in the background).
    *   `-p 8080:8080`: Maps port 8080 of the host VM to port 8080 of the container.
    *   `--name my-zenml-server`: Assigns a convenient name to your running container.
    *   `-v zenml_server_data:/zenml/.zenconfig`: **Important for persistence!** This creates a Docker named volume called `zenml_server_data` and mounts it to `/zenml/.zenconfig` inside the container. The Dockerfile specifies `ZENML_CONFIG_PATH=/zenml/.zenconfig`, so this is where ZenML server will store its data (like the default SQLite database, configurations, etc.). Without this, if you stop and remove the container, all ZenML server data will be lost.
    *   `zenml-server-custom`: The name of the image you built in the previous step.
  
  or

  `sudo docker run -it -d -p 8080:8080 --name zenml zenmldocker/zenml-server`
  restarting
  `sudo docker start zenml`

1.  **Verify the Server is Running:**
    *   Check if the container is running:
        ```bash
        sudo docker ps
        ```
        You should see `my-zenml-server` in the list with port `0.0.0.0:8080->8080/tcp`.
    *   Check the container logs (especially if it's not running or you encounter issues):
        ```bash
        sudo docker logs my-zenml-server
        ```
        Or for live logs:
        ```bash
        sudo docker logs -f my-zenml-server
        ```

**Phase 3: Connect your ZenML Client**

1.  **On your local machine** (or wherever you use the ZenML CLI):
    *   Ensure your ZenML client is installed and up-to-date (`pip install --upgrade zenml`).
    *   Connect to your new ZenML server:
        ```bash
        zenml connect --url http://YOUR_AZURE_VM_PUBLIC_IP:8080
        ```
        Replace `YOUR_AZURE_VM_PUBLIC_IP` with the actual public IP address of your Azure VM.
        *   The Dockerfile you provided doesn't seem to set up specific initial admin credentials directly within the Dockerfile build process itself. The ZenML server application running inside will handle its own user management. When you first connect, ZenML might guide you or use default credentials (often `default` as username and an empty or specific default password for the very first user, but refer to the ZenML version's documentation for specifics if the connect command prompts for more).

**Running ZenML and MLflow on the Same Machine:**

*   If your MLflow instance is also running on this same Azure VM, ensure there are no port conflicts. MLflow typically uses port 5000 by default, and your ZenML server is now on 8080, so they should coexist fine.
*   You will configure ZenML to use your MLflow instance by setting up an MLflow experiment tracker component in your ZenML stack, pointing to `http://YOUR_AZURE_VM_PUBLIC_IP:5000` (or `http://localhost:5000` if ZenML pipelines run on the same VM and can access MLflow via localhost).

This setup provides you with a self-hosted ZenML server using the Dockerfile you've prepared. Remember to manage your Azure VM's security (OS updates, etc.) and consider backup strategies for the `zenml_server_data` Docker volume if it contains critical data.