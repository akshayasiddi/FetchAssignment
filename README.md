# FetchAssignment

### Prerequisites

Before running the Streamlit app using Docker, make sure you have the following software installed:

- Docker: [Docker Installation Guide](https://docs.docker.com/get-docker/)

### Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/akshayasiddi/FetchAssignment.git
   ```

2. Build the Docker image:

   ```bash
   docker build -t streamlit-prediction-app .
   ```

3. Run the Docker container:

   ```bash
   docker run -p 8080:8501 streamlit-prediction-app
   ```

   - The `-p` flag maps port 8501 from the container to your host machine, allowing you to access the app in your browser.

4. Access the Streamlit app in your web browser:

   Open your web browser and navigate to [http://localhost:8080](http://localhost:85080).
   
### Alternate
Alternatively, you can access the app online at [https://scanned-receipts-prediction.streamlit.app/](https://scanned-receipts-prediction.streamlit.app/).
