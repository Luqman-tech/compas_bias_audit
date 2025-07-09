# COMPAS Bias Audit Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AIF360](https://img.shields.io/badge/AIF360-IBM-orange.svg)](https://github.com/Trusted-AI/AIF360)

## Overview

This project conducts a comprehensive bias audit of the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) recidivism prediction algorithm. COMPAS is widely used in the US criminal justice system to assess the likelihood of reoffending, but has been criticized for racial bias.

## 🎯 Objectives

- **Detect and quantify** racial bias in COMPAS risk scores
- **Visualize disparities** in false positive rates across racial groups
- **Demonstrate bias mitigation** techniques using preprocessing methods
- **Provide actionable insights** for fairer algorithmic decision-making

## 📊 Key Findings

- **23% higher false positive rate** for African-American defendants
- **Disparate impact ratio of 0.65** (below 0.8 fairness threshold)
- **70% bias reduction** achieved through reweighting techniques
- **Maintained predictive accuracy** while improving fairness

## 🛠️ Technology Stack

- **Python 3.8+**
- **AI Fairness 360 (AIF360)** - IBM's bias detection and mitigation toolkit
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning metrics
- **matplotlib/seaborn** - Data visualization
- **numpy** - Numerical computing

## 📁 Project Structure

```
compas-bias-audit/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── compas_bias_audit.py     # Main analysis script
├── bias_report.md           # Detailed findings report
├── data/                    # Dataset (auto-downloaded)
├── outputs/                 # Generated visualizations
│   ├── bias_metrics.png
│   ├── confusion_matrices.png
│   └── fairness_comparison.png
├── notebooks/               # Jupyter notebooks (optional)
│   └── exploratory_analysis.ipynb
└── LICENSE                  # MIT License
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Luqman-tech/compas_bias_audit.git
cd compas_bias_audit
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Analysis
```bash
python compas_bias_audit.py
```

## 📋 Requirements

- **Python 3.8+** (recommended)
- All required Python packages are listed in [requirements.txt](requirements.txt). Install them with:

```bash
pip install -r requirements.txt
```

- For advanced bias mitigation algorithms, you may need optional dependencies:
    - `tensorflow` (for AdversarialDebiasing):
      ```bash
      pip install 'aif360[AdversarialDebiasing]'
      ```
    - `fairlearn` (for Reductions):
      ```bash
      pip install 'aif360[Reductions]'
      ```
    - `inFairness` (for SenSeI and SenSR):
      ```bash
      pip install 'aif360[inFairness]'
      ```

- If you encounter errors related to missing system libraries (especially on Windows), see the [AIF360 installation guide](https://aif360.readthedocs.io/en/latest/install.html#installation) for troubleshooting tips.

## 📈 Visualizations

The script generates several key visualizations:

1. **Recidivism Rates by Race** - Baseline comparison of actual recidivism rates
2. **False Positive Rate Disparities** - Shows prediction bias across racial groups
3. **Confusion Matrices** - Detailed breakdown of prediction accuracy by race
4. **Bias Metrics Summary** - Comprehensive fairness assessment dashboard

## 🔍 Bias Metrics Analyzed

- **Disparate Impact** - Ratio of favorable outcomes between groups
- **Statistical Parity Difference** - Difference in positive prediction rates
- **Equal Opportunity Difference** - Difference in true positive rates
- **Equalized Odds** - Difference in both true positive and false positive rates
- **Calibration** - Accuracy of probability estimates across groups

## 🛡️ Bias Mitigation Techniques

### Preprocessing
- **Reweighting** - Adjusts sample weights to balance outcomes
- **Disparate Impact Remover** - Removes correlation with protected attributes

### Postprocessing
- **Equalized Odds** - Adjusts predictions to equalize error rates
- **Calibrated Equalized Odds** - Maintains calibration while ensuring fairness

## 📊 Results Summary

| Metric | Before Mitigation | After Mitigation | Improvement |
|--------|-------------------|------------------|-------------|
| Mean Difference | 0.170 | 0.055 | 67.6% |
| Disparate Impact | 0.65 | 0.89 | 36.9% |
| FPR Difference | 0.115 | 0.034 | 70.4% |

## 🔧 Usage Examples

### Basic Analysis
```python
from compas_bias_audit import CompasAuditor

# Initialize auditor
auditor = CompasAuditor()

# Run bias analysis
results = auditor.analyze_bias()

# Generate visualizations
auditor.create_visualizations()
```

### Custom Bias Metrics
```python
# Calculate specific fairness metrics
disparate_impact = auditor.calculate_disparate_impact()
equal_opportunity = auditor.calculate_equal_opportunity()
```

## 📚 Documentation

- **[Bias Report](bias_report.md)** - Detailed findings and recommendations
- **[AIF360 Documentation](https://aif360.readthedocs.io/)** - IBM's fairness toolkit
- **[COMPAS Dataset Info](https://github.com/propublica/compas-analysis)** - Original ProPublica analysis

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Research** for the AIF360 toolkit
- **ProPublica** for the original COMPAS investigation
- **AI Fairness research community** for methodology and best practices

## 📞 Contact

- **Author**: Philip Kisaih Iringo
- **Email**: philipiringo@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/philipkisaihiringo/
- **GitHub**: https://github.com/Luqman-tech

## 🔬 Future Work

- [ ] Implement additional bias mitigation algorithms
- [ ] Add support for intersectional bias analysis
- [ ] Create interactive dashboard for bias exploration
- [ ] Extend analysis to other criminal justice datasets
- [ ] Develop fairness-aware machine learning models

## 📖 References

1. Angwin, J., et al. (2016). "Machine Bias." ProPublica.
2. Bellamy, R. K., et al. (2018). "AI Fairness 360: An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias."
3. Barocas, S., Hardt, M., & Narayanan, A. (2019). "Fairness and Machine Learning."

---

**⚠️ Important Note**: This project is for educational and research purposes. Results should not be used to make decisions about real individuals without proper validation and ethical review.
