# 📊 Portfolio Management and Diversification Strategies

## 📝 Overview
This research project explores **portfolio diversification strategies**, focusing on four key methods:
- **Equal-Weight Portfolio (ENB)** – Each asset receives the same weight.
- **Equal-Risk Contribution (ERC)** – Allocates weights based on risk contribution to the portfolio.
- **Mean-Variance Optimization (MVO)** – Maximizes return for a given level of risk using Markowitz’s theory.
- **Rao's Quadratic Entropy (RQE)** – Incorporates entropy measures for diversification.

The project involves mathematical modeling, optimization algorithms, and empirical analysis using **R** and **Python**.

## 🔍 Research Goals
- Compare the risk-return trade-offs of different portfolio construction methods.
- Assess how diversification affects portfolio stability and performance.
- Implement and visualize portfolio allocation strategies using **real-world financial data**.

## 📂 Repository Structure
```
Portfolio_Management/
│── data/                 # Contains financial datasets
│── src/              
│── README.md             # Project documentation
│── requirements.txt      # Python dependencies
```

## ⚙️ Installation & Setup

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+** 
- Required Python libraries (install via `requirements.txt`)


### Python Setup
```bash
git clone https://github.com/AdvancedUno/Portfolio_Management.git
cd Portfolio_Management
pip install -r requirements.txt
```

## 🚀 Running the Analysis

### Running Python Scripts
```bash
python main.py
```


## 📊 Results & Insights
- **Risk-return comparisons:** RQE portfolios often outperform traditional methods in terms of stability.
- **Impact of correlation structures:** Highly correlated assets reduce ERC effectiveness.
- **Entropy-based diversification:** RQE provides a more balanced allocation by considering overall portfolio diversity.

## 📚 References
- Markowitz, H. (1952). *Portfolio Selection*, The Journal of Finance.
- Rao, C.R. (1982). *Diversity and Quadratic Entropy in Portfolio Selection*.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). *ERC Portfolio Construction*.

## 🏆 Acknowledgments
Special thanks to **Dr. White** for guidance on portfolio theory and optimization methodologies.
