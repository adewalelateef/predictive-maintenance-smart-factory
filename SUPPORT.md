# 💬 Support & Resources

Welcome! If you need help with the Smart Factory Predictive Maintenance project, here's where to find it.

## 🆘 Getting Help

### Quick Answers
1. **Check the [README](README.md)** - Start here for quick start and overview
2. **Search [Existing Issues](https://github.com/adewalelateef/predictive-maintenance-smart-factory/issues)** - Your question may already be answered
3. **Review [Discussions](https://github.com/adewalelateef/predictive-maintenance-smart-factory/discussions)** - Community Q&A

### Ask a Question
- 💬 **[Open a Discussion](https://github.com/adewalelateef/predictive-maintenance-smart-factory/discussions)** - Best for questions and general support
- 📧 **[Contact Maintainer](https://github.com/adewalelateef)** - For critical issues

### Report a Problem
- 🐛 **[Report a Bug](https://github.com/adewalelateef/predictive-maintenance-smart-factory/issues/new?assignees=adewalelateef&labels=bug&template=bug_report.md)** - Use the bug report template
- ✨ **[Request a Feature](https://github.com/adewalelateef/predictive-maintenance-smart-factory/issues/new?assignees=adewalelateef&labels=enhancement&template=feature_request.md)** - Use the feature request template

---

## 📚 Documentation

### Project Documentation
- [README.md](README.md) - Project overview and quick start
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history and roadmap

### Technical Guides
- [Notebooks](notebooks/) - Jupyter notebooks for ML pipeline
  - `00_setup.ipynb` - Environment setup
  - `01_eda_and_setup.ipynb` - Exploratory data analysis
  - `02_feature_engineering.ipynb` - Feature creation
  - `03_modeling.ipynb` - Model training & optimization

### Visual Guides
- [System Architecture](assets/architecture.svg) - High-level system design
- [Feature Importance](assets/feature_importance.svg) - What drives predictions
- [Quick Start](assets/quick_start.svg) - Getting started in 5 steps
- [Model Performance](assets/model_performance.svg) - Metrics & confusion matrix

---

## 🎓 Learning Resources

### Getting Started with the Project
1. **Read** the [README.md](README.md)
2. **Clone** the repository
3. **Install** dependencies: `pip install -r requirements.txt`
4. **Run** the dashboard: `streamlit run dashboard/app.py`
5. **Explore** the different pages

### Understanding the ML Model
- Learn about **XGBoost**: [XGBoost Docs](https://xgboost.readthedocs.io/)
- Understand **SHAP**: [SHAP Documentation](https://shap.readthedocs.io/)
- Study **Hyperparameter Tuning**: [Optuna Docs](https://optuna.readthedocs.io/)

### Learning Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Predictive Maintenance Concepts
- [Predictive Maintenance Overview](https://www.ibm.com/cloud/learn/predictive-maintenance)
- [Industry 4.0 Fundamentals](https://www.coursera.org/learn/iot-smart-factory)
- [ML for Manufacturing](https://www.linkedin.com/learning/machine-learning-for-manufacturing)

---

## 🐛 Troubleshooting

### Common Issues

#### Model Not Loading
```
❌ Model not found error
```
**Solution:**
- Check that `src/models/final_xgb_model.pkl` exists
- Ensure you've trained the model using `03_modeling.ipynb`
- Verify the MODEL_PATH in `dashboard/utils/constants.py`

#### Port Already in Use
```
❌ Port 8501 already in use
```
**Solution:**
```bash
# Find process using port
netstat -ano | findstr :8501  # Windows
lsof -i :8501                 # macOS/Linux

# Kill the process or use different port
streamlit run dashboard/app.py --server.port 8502
```

#### Dependencies Installation Fails
```
❌ Error installing requirements
```
**Solution:**
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### Docker Build Fails
```
❌ Docker build error
```
**Solution:**
```bash
# Clean up Docker system
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t smart-factory-ml .
```

### Performance Issues

#### Dashboard is Slow
- **Increase Streamlit cache time**: Update `@st.cache_resource` decorators
- **Reduce data size**: Use a subset of data for testing
- **Upgrade hardware**: More RAM/CPU improves performance

#### Memory Issues
- **Reduce batch size**: Modify data loading parameters
- **Optimize SHAP computation**: Use background data subset
- **Allocate more Docker memory**: `docker run -m 4g ...`

---

## 🤝 Getting Involved

### Ways to Contribute
- 🐛 **Report bugs** and help us squash them
- ✨ **Suggest features** you'd like to see
- 📝 **Improve documentation** with examples or clarifications
- 🔄 **Submit pull requests** with improvements
- 💬 **Participate in discussions** to help other users

### Development Setup
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup instructions.

---

## 📊 Community Stats

- ⭐ Stars: [View on GitHub](https://github.com/adewalelateef/predictive-maintenance-smart-factory)
- 🍴 Forks: [Fork this repo](https://github.com/adewalelateef/predictive-maintenance-smart-factory/fork)
- 👥 Contributors: [View contributors](https://github.com/adewalelateef/predictive-maintenance-smart-factory/graphs/contributors)
- 📋 Open Issues: [View issues](https://github.com/adewalelateef/predictive-maintenance-smart-factory/issues)

---

## 🔗 Useful Links

### Official Resources
- 🌐 [Project Repository](https://github.com/adewalelateef/predictive-maintenance-smart-factory)
- 🔧 [GitHub Issues](https://github.com/adewalelateef/predictive-maintenance-smart-factory/issues)
- 💬 [GitHub Discussions](https://github.com/adewalelateef/predictive-maintenance-smart-factory/discussions)
- 👤 [GitHub Profile](https://github.com/adewalelateef)

### External Resources
- 🏢 [Streamlit](https://streamlit.io/)
- 🤖 [XGBoost](https://xgboost.readthedocs.io/)
- 📊 [SHAP](https://shap.readthedocs.io/)
- 🔍 [Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

---

## 📬 Contact

- **GitHub Issues**: [Open an issue](https://github.com/adewalelateef/predictive-maintenance-smart-factory/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/adewalelateef/predictive-maintenance-smart-factory/discussions)
- **Author**: [Adewale Lateef](https://github.com/adewalelateef)

---

## ⭐ Show Your Support

If you find this project helpful:
- ⭐ Star the repository
- 🔄 Share it with others
- 💬 Leave feedback and suggestions
- 🤝 Contribute improvements

Thank you for using Smart Factory Predictive Maintenance! 🙏
