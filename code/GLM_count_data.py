import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sns.set()
from statsmodels.graphics.gofplots import ProbPlot
plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

data = pd.read_csv('4_Parastism_Fig_6.csv',  delimiter=';')
data.head()

sns.countplot(x="Parasitized", hue="Treatment", data=data)
plt.savefig("Histogram of the count.pdf")

X = data[['Fruit', 'Treatment']]
y = data['Parasitized']

import statsmodels.formula.api as smf
model_fit = smf.ols(formula="Parasitized ~ C(Treatment)", data=data).fit()
res=model_fit.resid

fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios':[2,1]})

sns.distplot(res, ax=axes[0])
axes[0].set_title('Distribution of the residuals')

sm.qqplot(res, fit=True, line="45", ax=axes[1])
axes[1].set_title('Normal qqplot')
plt.savefig("Distribution and normal QQ plot residuals .pdf")

plot_lm_3 = plt.figure()
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(i,
                               xy=(model_fitted_y[i],
                                   model_norm_residuals_abs_sqrt[i]));
    
plt.savefig("scale-location plot.pdf")

pois_sim1 = np.random.poisson(1, 10000)
pois_sim5 = np.random.poisson(5, 10000)
pois_sim10 = np.random.poisson(10, 10000)
pois_sim15 = np.random.poisson(15, 10000)
pois_sim20 = np.random.poisson(20, 10000)
pois_sim30 = np.random.poisson(30, 10000)

# par(las=1,mar=c(4,4,3,1))
# layout(matrix(c(1,2,3,3), 2, 2, byrow = TRUE))
# plot(train_xts,main='Traing Data')
# plot(test_xts,main='Testing Data')
# plot(as.zoo(whole_xts), screens=1,main='Whole Data',ylab='',xlab='')

# f, axes = plt.subplots(2, 3, figsize=(7, 7), sharex=True)
fig, axes = plt.subplots(2, 3, figsize=(9, 7), sharex=True)

sns.distplot(pois_sim1, kde=False, ax=axes[0, 0])
axes[0, 0].set_title("lambda=1")

sns.distplot(pois_sim5, kde=False, ax=axes[0, 1])
axes[0,1].set_title("lambda=5")

sns.distplot(pois_sim10, kde=False, ax=axes[0, 2])
axes[0,2].set_title("lambda=10")

sns.distplot(pois_sim15, kde=False, ax=axes[1, 0])
axes[1,0].set_title("lambda=15")

sns.distplot(pois_sim20, kde=False, ax=axes[1, 1])
axes[1,1].set_title("lambda=20")

sns.distplot(pois_sim15, kde=False, ax=axes[1, 2])
axes[1,2].set_title("lambda=30")

plt.setp(axes, yticks=[])
plt.tight_layout()
# plt.show()

plt.savefig("histogram poisson dist.pdf")

from patsy import dmatrices
import statsmodels.api as sm

formula = """Parasitized ~ C(Treatment)"""
response, predictors = dmatrices(formula, data, return_type='dataframe')
po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
print(po_results.summary())

formula = """Parasitized ~ C(Treatment)"""
response, predictors = dmatrices(formula, data, return_type='dataframe')
po_results = sm.GLM(response, predictors, family=sm.families.NegativeBinomial()).fit()
print(po_results.summary())

modpoiss =smf.poisson(formula,data).fit()
print(modpoiss.summary())

modNB =smf.negativebinomial(formula,data).fit()
print(modNB.summary())

stats.probplot(modpoiss.resid, dist='poisson', sparams=(2.4,), plot=plt)
plt.show()

stats.probplot(modNB.resid, dist='nbinom', sparams=(2.15,0.4), plot=plt)
plt.show()

print(modNB.get_margeff('mean').summary()) 