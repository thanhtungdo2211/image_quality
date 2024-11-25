# Image Quality Analysis

This repository contains a Python-based solution for analyzing image quality. The project is structured into three main directories: `src`, `test`, and `utils`.

## Directory Structure

- `src/`: Contains the main Python scripts for issue analysis and scoring.
- `test/`: Contains Jupyter notebooks for testing the functionality of the scripts in `src/`.
- `utils/`: Contains utility scripts used by the main scripts and tests.

## Key Features

- **Issue Analysis**: The script in [src/issue_analysis.py](src/issue_analysis.py) provides functionality for analyzing various issues in video quality.
- **Issue Scoring**: The script in [src/issue_score.py](src/issue_score.py) provides functionality for scoring the severity of detected issues.
- **Visualization**: The script in [src/visualize.py](src/visualize.py) provides functionality for visualizing the results of the issue analysis and scoring.

## Usage

To use this project, you will need to install the required Python packages listed in `requirements.txt`. Then, you can run the scripts [main.py](main.py) to analyze and score selected specific image quality issues.

## Testing

The `test/` directory contains Jupyter notebooks that demonstrate how to use the scripts in `src/`. For example, [test/camera_tampering.ipynb](test/camera_tampering.ipynb) shows how to use the scripts to detect and score camera tampering issues.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the MIT license.
