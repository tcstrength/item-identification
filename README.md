# Project Setup Guide

This README provides instructions for setting up the Python environment for this project using VSCode.

## Setup

### Prerequisites

- VSCode installed on your system
- Python 3.9 installed
- `uv` package manager installed, visit this for more information: [click here](https://docs.astral.sh/uv/getting-started/installation/)

### Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/tcstrength/item-identification.git
   cd item-identification
   ```

2. **Set up Python environment with uv**

   Create and activate a virtual environment:

   ```bash
   uv venv
   ```

   On Windows, activate the environment:
   ```bash
   .venv\Scripts\activate
   ```

   On macOS/Linux, activate the environment:
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   uv pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the root directory of the project:

   ```bash
   touch .env
   ```

   Add the following environment variables to the `.env` file:

   ```
   LABEL_STUDIO_URL=<your-label-studio-url>
   LABEL_STUDIO_API_KEY=<your-api-key>
   LABEL_STUDIO_TEMP_DIR=local/temp
   LABEL_STUDIO_PROJECT_MAPPING={"train":1,"validation":7,"test":3}
   ```

5. **Configure VSCode**

   - Open the project in VSCode
   - Install recommended extensions (Python, Python Environment Manager)
   - Select the Python interpreter:
     - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
     - Type "Python: Select Interpreter"
     - Choose the interpreter from your `.venv` directory

### Additional Notes

- Make sure to never commit your `.env` file to version control
- If you update dependencies, update requirements.txt with: `uv pip freeze > requirements.txt`

## References:
- Selective Search for Object Detection: https://www.geeksforgeeks.org/selective-search-for-object-detection-r-cnn/
- YOLOv12: https://docs.ultralytics.com/models/yolo12/#citations-and-acknowledgements
- Low Object Count: https://chatgpt.com/share/6804ca08-de5c-8001-9be8-402becd967b8
