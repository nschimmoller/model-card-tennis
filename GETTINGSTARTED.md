# Getting Started with Model Card Tennis

This guide will walk you through the steps of downloading and running the code from the Model Card Tennis repository on your local computer. This guide assumes that you do not have Python installed and that you are using either a Mac or Windows operating system.

## Step 1: Install Python

### Option A (Recommended): Install Anaconda

Anaconda is a popular distribution of Python that comes bundled with many scientific computing packages, including numpy and pandas, which are used in this repository. To download Anaconda, follow these steps:

1. Go to the [Anaconda download page](https://www.anaconda.com/products/individual#Downloads) and download the appropriate version for your operating system.

2. Follow the installation instructions for your operating system.

### Option B: Intall Python directly

Before you can run the code in this repository, you need to have Python installed on your computer. You can download Python for free from the official Python website: https://www.python.org/downloads/

For Mac users, download the macOS installer and run it to install Python on your computer. For Windows users, download the Windows installer and run it to install Python on your computer. During the installation process, make sure to select the "Add Python to PATH" option.

## Step 2: Clone the Repository

### Option A: Download Directly

1. Go to the [Model Card Tennis repository](https://github.com/nschimmoller/model-card-tennis) on GitHub.

2. Click the "Code" button and select "Download ZIP" to download the repository as a zip file.

3. Extract the contents of the zip file to a folder on your computer.

### Option B: Download via Git

1. Open a terminal or command prompt.

2. Navigate to the directory where you want to store the project.

3. Clone the repository using the following command:
   ```
   git clone https://github.com/nschimmoller/model-card-tennis.git
   ```
4. Once the repository is cloned, navigate into the project directory:
   ```
   cd model-card-tennis
   ```
This will download a copy of the repository to your local machine, which you can now run and modify as needed.

## Step 3: Install Required Packages

1. Open a terminal or command prompt window.

2. Navigate to the folder containing the extracted repository.

3. Run the following command to create a new Anaconda environment:

   ```
   conda create --name model-card-tennis python=3.8
   ```

4. Activate the environment by running the following command:

   - **On Windows:** `conda activate model-card-tennis`

   - **On Mac or Linux:** `source activate model-card-tennis`

5. Run the following command to install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Step 4: Run the Code

1. In the terminal or command prompt window, navigate to the folder containing the extracted repository.

2. Activate the Anaconda environment by running the following command:

   - **On Windows:** `conda activate model-card-tennis`

   - **On Mac or Linux:** `source activate model-card-tennis`

3. Run the following command to start the Jupyter Notebook:

   ```
   jupyter notebook
   ```

4. A web page should open in your default web browser showing the contents of the repository. Click on the "simulate_tennis_match.ipynb" file to open it.

5. Follow the instructions in the notebook to simulate a tennis match.

Congratulations! You have successfully downloaded and run the code from the Model Card Tennis repository.