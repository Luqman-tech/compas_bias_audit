# Local Setup Guide for COMPAS Bias Audit

## Prerequisites

- Python 3.7+ (recommended: Python 3.8-3.10)
- pip package manager
- Git (optional, for cloning repositories)

## Step 1: Create Virtual Environment

```bash
# Create a virtual environment
python -m venv compas_audit_env

# Activate virtual environment
# On Windows:
compas_audit_env\Scripts\activate
# On macOS/Linux:
source compas_audit_env/bin/activate
```

## Step 2: Install Required Packages

```bash
# Install core dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Install AI Fairness 360 (IBM's toolkit)
pip install aif360

# If you encounter issues with aif360, try:
pip install aif360[all]

# Additional dependencies that might be needed
pip install jupyter notebook ipython
```

## Step 3: Handle Common Installation Issues

### Issue 1: AIF360 Installation Problems
```bash
# If aif360 installation fails, try installing dependencies first:
pip install --upgrade pip setuptools wheel
pip install cython
pip install aif360

# For Apple Silicon Macs:
pip install --no-use-pep517 aif360
```

### Issue 2: Missing System Dependencies
```bash
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install build-essential python3-dev

# macOS (with Homebrew):
brew install gcc

# Windows: Install Visual Studio Build Tools
```

## Step 4: Download and Prepare the Code

1. **Save the Python code** to a file named `compas_bias_audit.py`
2. **Create project directory**:
```bash
mkdir compas_audit_project
cd compas_audit_project
```

## Step 5: Run the Program

```bash
# Make sure your virtual environment is activated
python compas_bias_audit.py
```

## Step 6: Alternative Setup with Jupyter Notebook

If you prefer running in Jupyter:

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Create new notebook and paste the code
```

## Troubleshooting Common Issues

### 1. COMPAS Dataset Download Issues
If the dataset doesn't download automatically:
```python
# Add this at the beginning of your script
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### 2. Matplotlib Display Issues
```bash
# For headless systems or display issues:
pip install matplotlib
export MPLBACKEND=Agg  # For non-interactive backend
```

### 3. Seaborn Style Warning
Update the seaborn style line in the code:
```python
# Change this line:
plt.style.use('seaborn-v0_8')
# To:
plt.style.use('seaborn-v0_8-whitegrid')  # or another available style
```

## Expected Output

When you run the program successfully, you should see:
- Dataset loading confirmation
- Bias metrics calculations
- Multiple visualization windows/plots
- Console output with numerical results
- Summary statistics

## File Structure
```
compas_bias_audit/
├── compas_bias_audit.py
├── compas_audit_env/         # Virtual environment
└── output/                   # Generated plots (optional)
```

## Performance Notes

- First run may take 2-5 minutes (dataset download)
- Subsequent runs will be faster (cached data)
- Memory usage: ~500MB-1GB
- CPU usage: Moderate during computation

## Saving Results

To save plots and results:
```python
# Add these lines to save plots
plt.savefig('bias_analysis_plots.png', dpi=300, bbox_inches='tight')

# Save metrics to file
with open('bias_metrics.txt', 'w') as f:
    f.write(f"Mean difference: {metric.mean_difference():.4f}\n")
    # Add other metrics...
```

## Next Steps

1. **Run the basic analysis** first to ensure everything works
2. **Modify parameters** to explore different scenarios
3. **Add custom visualizations** based on your needs
4. **Export results** for reporting purposes

## Getting Help

If you encounter issues:
- Check the [AIF360 documentation](https://aif360.readthedocs.io/)
- Review Python environment setup
- Ensure all dependencies are correctly installed
- Try running in a fresh virtual environment