# How to Contribute

Thank you for your interest in contributing to this project! This guide will walk you through setting up your local development environment, building the package, and updating the documentation.

## Local Development Environment Setup

To make changes to the package and upload it again, follow the steps specified below. These steps follow the Python Packaging User Guide.

### Step 1: Create a Virtual Environment

1. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:

    - On Unix/macOS:

        ```bash
        source venv/bin/activate
        ```

    - On Windows:

        ```bash
        venv\Scripts\activate.bat
        ```

### Step 2: Install Development Dependencies

1. Uncomment all the dependencies under `#Dev` in your `requirements.txt` file.

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Build the Package

To build the package locally, you can use the following commands:

1. **Install required tools**:

    ```bash
    pip install build
    pip install wheel
    ```

2. **Build the package**:

    ```bash
    python setup.py sdist bdist_wheel
    ```

### Step 4: Upload the Package with Twine

To upload the built package to a repository like TestPyPI or PyPI, follow these steps:

1. **Build the package (again)**:

    ```bash
    python setup.py sdist bdist_wheel
    ```

2. **Upload using Twine**:

    ```bash
    pip install twine
    twine upload dist/*
    ```

## Viewing the Documentation

To view the documentation locally:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-repo/melanoma-segmentation-and-classification.git
    cd melanoma-segmentation-and-classification
    ```

2. **Open the HTML file**:

    Navigate to the documentation output and open it in your browser:

    ```bash
    docs/build/html/melanoma_segmentation.html
    ```

## Updating the Sphinx Documentation

To update the Sphinx documentation, follow these steps:

1. **Delete all `.rst` files except `index.rst`**:

    ```bash
    find docs/source/ -type f ! -name 'index.rst' -delete
    ```

2. **Regenerate `.rst` files using `sphinx-apidoc`**:

    ```bash
    sphinx-apidoc -o docs/source/ melanoma_segmentation/
    ```

3. **Rebuild the documentation**:

    ```bash
    cd docs
    sphinx-build -b html source build/html
    ```

This process will update your documentation to reflect any changes made to your package.
