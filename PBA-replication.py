#!/usr/bin/env python
# coding: utf-8

# # _PROJECT_: Prediction Error $-$ Belief replication 

# In[1]:


# Core analysis packages

from pandas import ExcelWriter
from pandas import ExcelFile
from statsmodels.formula.api import ols
import pingouin as pg
import researchpy

import numpy as np
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import anova
from patsy import dmatrices
import bff
# Plotting packages
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns 
sns.set(style="ticks", color_codes=True)
sns.set_style("white")
sns.set_style({'xtick.bottom': True, 'ytick.left': True})
colorref = ["gray", "royalblue", "crimson", "goldenrod", "mediumorchid", "seagreen"]

# iPython magic commands
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIG_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title


# In[2]:


def ttest_ind(x1, x2, equivar=False, alpha=0.05, printres=False):
    n1 = len(x1)
    M1 = np.mean(x1)
    s1 = np.std(x1, ddof=1)
    n2 = len(x2)
    M2 = np.mean(x2)
    s2 = np.std(x2, ddof=1)
    
    # t-test
    [t, p] = stats.ttest_ind(x1, x2, equal_var=equivar)
    # cohen's d
    dof = n1 + n2 - 2
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / dof)
    d = np.abs(M1 - M2) / sp
    # degrees of freedom
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    # confidence intervals (M1 - M2) ± ts(M1 - M2)
    se = np.sqrt(sp**2/n1 + sp**2/n2)
    CI = (M1 - M2) + np.array([-1,1])*stats.t.ppf(1-alpha/2, df, loc=0, scale=1)*se

    res = (t, df, p, d, CI[0], CI[1])
    if printres:
        print("t = %.5f, df = %.5f, p = %.5f, d = %.5f, CI = (%.5f, %.5f)" % res)
    else:
        return res


# ## 1 $-$ Reshape data matrix into data column

# In[3]:


# Load file (from same directory as the notebook)
df = pd.read_excel(os.path.expanduser("~/Desktop/PBArep.xlsx"), index_col=0)
df.head()


# #### 1a $-$ Unpack data from dataframe
# In general, this code will change from project to project, based on data labels, partitions, etc.
# 
# We want to use the row/column labels printed in the header above to extract data (typically matrices) and data labels (typically vectors). In this case, participants are rows and items are columns, which means participant-based labels are columns and item-based labels are rows.

# In[4]:


# Extract rows with df.loc["target",:], only grabbing data corresponding to items
nitemlabel = 3
overunder = df.loc["Over_Under", "1N":"12R"]
supportagainst = df.loc["Support_Against", "1N":"12R"]
itemideology = df.loc["Dem_Rep_Ind", "1N":"12R"]
itemnum = np.arange(itemideology.size)
# Additional info (maximum achievable PE)
maxPE = np.array([10,9,7,6,8,11,8,6,11,10,9,7,9,7,10,6,8,11,11,8,6,9,10,7,7,9,10,8,11,6,11,8,6,7,10,9])
# Then remove these rows from the dataframe and continue
dftrim = df.iloc[nitemlabel:, :]

# Extract columns with df.target or df["target"] or df.loc[:,"target"]
date = dftrim.loc[:, "Date"]
study = dftrim.loc[:, "study"]
cond = dftrim.loc[:, "cond"]
partideology = dftrim.loc[:, "party"]
#resist = dftrim.loc[:, "BeliefResistance"]
#president = dftrim.loc[:, "SupportTrump"]
#partystr = dftrim.loc[:, "PartyStrength"]
#vote = dftrim.loc[:, "votereg"]
partnum = np.arange(partideology.size)

# Extract data with df.loc[:,"datastart":"dataend"]
deltaB = dftrim.loc[:, "1N":"12R"]
signedPE = dftrim.loc[:, "1N.1":"12R.1"]
PE = signedPE.abs().astype(int)
Bpre = dftrim.loc[:, "1N.2":"12R.2"]
sz = deltaB.shape
print(sz)


# #### 1b $-$ Repack data into new dataframe in column-form
# First, we expand every piece of data unpacked above to match the shape of the data. Then, we reshape all of them into columns and combine into a new dataset ready for analysis.

# In[5]:


names = ["deltaB", "signedPE", "PE", "Bpre", "overunder", "supportagainst", "itemideology", "maxPE", "itemnum",          "date", "study", "cond", "partideology", "partnum"]
data_packed = np.broadcast_arrays(deltaB, signedPE, PE, Bpre, overunder, supportagainst, itemideology, maxPE, itemnum)
cols = [np.tile(a, (sz[1],1)).T for a in [date, study, cond, partideology, partnum]]
data_packed += cols

data_unpacked = np.vstack([np.reshape(a, (1,-1), order="C") for a in data_packed]).T
DATA = pd.DataFrame(data=data_unpacked, columns=names)


# #### 1c $-$ Rational update direction
# 
# "Rational" update reflects a resigning of each belief change such that the "rational" direction is always positive. If a participant has a huge underestimated PE of evidence that refutes the belief, for example, the rational response is to decrease the belief; in data analysis terms, we multiply this belief change by $-1$:
# 
# $$\textsf{Rational }\Delta B = \textsf{sign}({Support/Against})\cdot\textsf{sign}({Over/Under})\cdot\textsf{sign}({PE})\cdot\Delta B$$

# In[6]:


# We add a small positive number to prediction errors such that sign(0) = 1
RUD = np.sign(DATA["supportagainst"] - 0.5) * np.sign(DATA["overunder"] - 0.5) * np.sign(DATA["signedPE"] + 0.01)
RUD_control = np.sign(DATA["supportagainst"] - 0.5)
RUD[DATA["cond"]==2] = RUD_control[DATA["cond"]==2]
# Rational belief update
RdeltaB = DATA["deltaB"]*RUD
if "RdeltaB" in DATA:
    DATA["RdeltaB"] = RdeltaB
else:
    DATA.insert(0, "RdeltaB", RdeltaB)
# Rational update interval
RpreI = np.abs((RUD+1)*50 - DATA["Bpre"])
if "RpreI" in DATA:
    DATA["RpreI"] = RpreI
else:
    DATA.insert(5, "RpreI", RpreI)
if "RpreI" in DATA:
    DATA["RpreI"] = RpreI
else:
    DATA.insert(5, "RpreI", RpreI)

DATA.head()


# ## 3 $-$ Plot belief vs. PE
# First, we transform the raw PEs into bins in a new column of the dataframe. Then, we can use seaborn to make simple barplots.

# In[7]:


bins = [[0],[1,2,3,4,5],[6,7,8,9,10,11]]
binedges = [a[0] - 0.5 for a in bins] + [bins[-1][-1] + 0.5]
# Digitize the prediction 
binnedPE = np.digitize(DATA["PE"], binedges)-1
# Force control condition to be bin "3" (the rightmost bin)
binnedPE[DATA["cond"]==2] = 3
DATA.insert(3,"binnedPE",binnedPE)


# In[8]:


makeslargePEs = np.unique(np.array(DATA.loc[DATA["binnedPE"]==2, "partnum"]))
canmakelargePEs = np.unique(np.array(DATA.loc[DATA["cond"]==1, "partnum"]))
print(np.setdiff1d(canmakelargePEs, makeslargePEs))


# In[9]:


DATA.head()


# In[12]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
get_ipython().run_line_magic('R', 'library(lmerTest)')


# In[13]:


get_ipython().run_line_magic('Rpush', 'DATA')


# In[14]:


anovaDATA = DATA.loc[DATA["cond"]==1, :]

get_ipython().run_line_magic('Rpush', 'anovaDATA')


# In[15]:


get_ipython().run_cell_magic('R', '', '\nanova(lmer(RdeltaB ~ PE*partideology + (1|partnum) + (1|itemnum), data=anovaDATA))')


# In[16]:


get_ipython().run_cell_magic('R', '', '\nanova(lmer(RdeltaB ~ itemideology*binnedPE*partideology + (1|partnum), data=anovaDATA))')


# #### Main effect

# In[17]:


#DATA.groupby(['binnedPE']).size() / (620*36)
DATA.groupby(['partideology', 'itemideology', 'binnedPE']).size() / (12 * 296)


# In[18]:


# Group and average data within participant and PE bin
AVDATA = DATA.groupby(["partnum"], as_index=False).mean()
# Plotting
fig, ax = plt.subplots(1,2, figsize=(6,4.5), gridspec_kw={"width_ratios": [1,2.125]}, sharey=True)
cust_palette = [sns.light_palette("gray", n_colors=len(bins))[1]] + [np.array([1, 1, 1, 0])]
sns.barplot(x="cond", y="RdeltaB", data=AVDATA,             palette=cust_palette, linewidth = [0,2], edgecolor=".2", errcolor=".2",             ax=ax[0]) 
sns.despine()
ax[0].set_xlabel("");
ax[0].set_xticklabels(["PE","Control"], rotation=45);
ax[0].set_ylabel("Rational belief update");
# Significance testing
datahandles = np.array([L.get_xydata() for L in ax[0].get_lines()])
_, p = stats.ttest_ind(AVDATA.loc[AVDATA["cond"]==1, "RdeltaB"],                        AVDATA.loc[AVDATA["cond"]==2, "RdeltaB"], equal_var=False)
bff.barplot_annotate_brackets(ax[0], 0, 1, p, datahandles, dh=0.05, fs=14)
plt.tight_layout()
bff.panellabel(ax[0], "A")



# Group and average data within participant and PE bin
AVDATA = DATA.groupby(["partnum", "binnedPE"], as_index=False).mean()
# Plotting
cust_palette = sns.light_palette("gray", n_colors=len(bins)) + [np.array([1, 1, 1, 0])]
sns.barplot(x="binnedPE", y="RdeltaB", data=AVDATA,             palette=cust_palette, linewidth = [0]*len(bins) + [2], edgecolor=".2", errcolor=".2",             ax=ax[1]) 
sns.despine()
ax[1].set_xlabel("");
ax[1].set_xticklabels(["No PE","Small PE","Large PE","Control"], rotation=45);
ax[1].set_ylabel("");
# Significance testing
datahandles = np.array([L.get_xydata() for L in ax[1].get_lines()])
_, p = stats.ttest_ind(AVDATA.loc[AVDATA["binnedPE"]==2, "RdeltaB"],                        AVDATA.loc[AVDATA["binnedPE"]==3, "RdeltaB"], equal_var=False)
bff.barplot_annotate_brackets(ax[1], 2, 3, p, datahandles, dh=0.05, fs=14)
plt.tight_layout()
bff.panellabel(ax[1], "B")

#plt.savefig('/Users/mada/Desktop/fig2rep.tif', dpi=300, format="tiff")


# #### Main effect, split by participant $\times$ item ideology

# In[19]:


# Declare groups over which to split here:
grp1 = "partideology"
grp2 = "itemideology"

fig, ax = plt.subplots(len(DATA[grp1].unique()), len(DATA[grp2].unique()), figsize=(9,6), sharey=True)
colors = ["royalblue", "crimson", "goldenrod"]

# Split dataframe into groups, where grpDATA is a subsampled dataframe of DATA with one split's data at a time
svals = []
for name, grpDATA in DATA.groupby([grp1, grp2]):
    name = np.array(name, dtype=int) - 1
    cax = ax[name[0],name[1]]
    # Average the split's items within participants
    AVDATA = grpDATA.groupby(["partnum", "binnedPE"], as_index=False).mean()
    
    # Plotting
    cust_palette = sns.light_palette(colors[name[-1]], n_colors=len(bins)) + [np.array([1, 1, 1, 0])]
    sns.barplot(x="binnedPE", y="RdeltaB", data=AVDATA,                 palette=cust_palette, linewidth = [0]*len(bins) + [2], edgecolor=".2", errcolor=".2",                 ax=cax) 
    sns.despine()
    cax.set_xlabel("");
    if name[0] == len(DATA[grp1].unique())-1:
        cax.set_xticklabels(["No PE","Small PE","Large PE","Control"], rotation=45);
    else:
        cax.set_xticklabels("");
    if name[1] == 0:
        cax.set_ylabel("Rational belief update");
    else:
        cax.set_ylabel("");
    # Significance testing
    datahandles = np.array([L.get_xydata() for L in cax.get_lines()])
    res = ttest_ind(AVDATA.loc[AVDATA["binnedPE"]==2, "RdeltaB"],                     AVDATA.loc[AVDATA["binnedPE"]==3, "RdeltaB"])
    svals += [res]
    bff.barplot_annotate_brackets(cax, 2, 3, res[2], datahandles, dh=0.05, fs=14)

plt.tight_layout()
#plt.savefig('/Users/mada/Desktop/fig4rep.tif', dpi=300, format="tiff")


# In[20]:


reference = ["", "DEM", "REP", "NEU"]
for name, grpDATA in DATA.groupby(["partideology","itemideology"]):
    print("Participant: ", reference[int(name[0])], " Item: ", reference[int(name[1])])
    print("MEAN: ", list(grpDATA.groupby(["cond","binnedPE"]).mean()["RdeltaB"]),           "\n",           "STD:", list(grpDATA.groupby(["cond","binnedPE"]).std()["RdeltaB"]))
    print("t = %.5f, df = %.5f, p = %.5f, d = %.5f, CI = (%.5f, %.5f)\n" % svals[3*int(name[0]-1) + int(name[1]-1)])


# #### Miscellaneous comparisons (between-panel t-tests) 

# In[21]:


AVDATA = DATA.groupby(["partnum", "binnedPE","partideology", "itemideology"], as_index=False).mean()
## ##
idx1 = np.logical_and(np.logical_and(AVDATA["partideology"]==1, AVDATA["itemideology"]==1), AVDATA["binnedPE"]==2)
idx2 = np.logical_and(np.logical_and(AVDATA["partideology"]==2, AVDATA["itemideology"]==1), AVDATA["binnedPE"]==2)
res = ttest_ind(AVDATA.loc[idx1,"RdeltaB"], AVDATA.loc[idx2,"RdeltaB"])
print("DEM vs REP participants, DEM items")
print("t = %.5f, df = %.5f, p = %.5f, d = %.5f, CI = (%.5f, %.5f)\n" % res)
## ##
idx1 = np.logical_and(np.logical_and(AVDATA["partideology"]==1, AVDATA["itemideology"]==2), AVDATA["binnedPE"]==2)
idx2 = np.logical_and(np.logical_and(AVDATA["partideology"]==2, AVDATA["itemideology"]==2), AVDATA["binnedPE"]==2)
res = ttest_ind(AVDATA.loc[idx1,"RdeltaB"], AVDATA.loc[idx2,"RdeltaB"])
print("DEM vs REP participants, REP items")
print("t = %.5f, df = %.5f, p = %.5f, d = %.5f, CI = (%.5f, %.5f)\n" % res)
## ##
idx1 = np.logical_and(np.logical_and(AVDATA["partideology"]==1, AVDATA["itemideology"]==1), AVDATA["binnedPE"]==2)
idx2 = np.logical_and(np.logical_and(AVDATA["partideology"]==1, AVDATA["itemideology"]==2), AVDATA["binnedPE"]==2)
res = ttest_ind(AVDATA.loc[idx1,"RdeltaB"], AVDATA.loc[idx2,"RdeltaB"])
print("DEM vs REP items, DEM participants")
print("t = %.5f, df = %.5f, p = %.5f, d = %.5f, CI = (%.5f, %.5f)\n" % res)
## ##
idx1 = np.logical_and(np.logical_and(AVDATA["partideology"]==2, AVDATA["itemideology"]==1), AVDATA["binnedPE"]==2)
idx2 = np.logical_and(np.logical_and(AVDATA["partideology"]==2, AVDATA["itemideology"]==2), AVDATA["binnedPE"]==2)
res = ttest_ind(AVDATA.loc[idx1,"RdeltaB"], AVDATA.loc[idx2,"RdeltaB"])
print("DEM vs REP items, REP participants")
print("t = %.5f, df = %.5f, p = %.5f, d = %.5f, CI = (%.5f, %.5f)\n" % res)


# ### Congruent vs. Incongruent belief updates

# In[23]:


AVDATA = DATA.groupby(["partnum", "binnedPE", "partideology", "itemideology"], as_index=False).mean()
AVDATA = AVDATA.loc[np.logical_or(AVDATA["binnedPE"]==2, AVDATA["binnedPE"]==3), :]
AVDATA = AVDATA.loc[np.logical_or(AVDATA["itemideology"]==1, AVDATA["itemideology"]==2), :]
AVDATA.insert(1, "congruent", np.array(np.equal(AVDATA["itemideology"], AVDATA["partideology"]), dtype=np.int))

#aov = pg.mixed_anova(dv='RdeltaB', within='binnedPE', between='congruent', data=AVDATA)

aov = pg.anova(dv='RdeltaB', between=["binnedPE","congruent"], data=AVDATA, detailed=True)
aov


# In[25]:


# Group and average data within participant and PE bin
AVDATA = DATA.loc[DATA["cond"]==1,:].groupby(["partnum", "PE"], as_index=False).mean()

# Plotting
fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.regplot(x="PE", y="RdeltaB", x_estimator=np.mean, data=AVDATA,             color="gray", ci=95, ax=ax)
sns.despine()
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
ax.set_xlabel("Prediction Error");
ax.set_ylabel("Rational belief update");

print(sm.OLS(AVDATA["RdeltaB"], sm.add_constant(AVDATA["PE"])).fit().summary())

plt.tight_layout()
#plt.savefig('/Users/mada/Desktop/fig1rep.tif', dpi=300, format="tiff")


# ## 4 $-$ Linear mixed effects model using ``statsmodels`` in Python
# 
# Let's build up the model and measure in-sample prediction RMSE as a performance measure for our evolving models. To begin, add a z-scored PE such that regression coefficients are comparable across categorical and linear predictors.

# In[26]:


DATAexp = DATA[DATA["cond"]==1]


# In[27]:


if "scaledPE" not in DATAexp:
    DATAexp.insert(4, "scaledPE", DATAexp[["PE"]].apply(stats.zscore))
map_dict = {1: "D", 2: "R", 3: "aN"}
if "pid" in DATAexp:
    DATAexp = DATAexp.drop("pid", axis=1)
    DATAexp = DATAexp.drop("iid", axis=1)
DATAexp.insert(12,"pid",DATAexp["partideology"].map(map_dict))
DATAexp.insert(10,"iid",DATAexp["itemideology"].map(map_dict))

DATAexp.head()


# ## 5 $-$ Linear mixed effects model using ``lme4`` in R
# Crossed random effects are not supported in ``statsmodels``. One workaround is to directly call ``lme4`` in R from the ipython environment. Let's launch an R environment:
# 
# Import the R package ``lme4`` (``RRuntimeWarning: Loading required package: Matrix`` is normal!) Also, for whatever reason the kernel will die if we load the ``lme4`` and ``lmerTest`` libraries in the same cell??

# In[28]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('load_ext', 'rpy2.ipython')
# %R library(lme4)


# In[29]:


get_ipython().run_line_magic('R', 'library(lmerTest)')


# - Push data to the R side (``FutureWarning: from_items is deprecated`` is normal)

# In[32]:


if "scaledRpreI" not in DATAexp:
    DATAexp.insert(7, "scaledRpreI", DATAexp[["RpreI"]].apply(stats.zscore))
map_dict = {1: "D", 2: "R", 3: "aN"}
if "pid" in DATAexp:
    DATAexp = DATAexp.drop("pid", axis=1)
    DATAexp = DATAexp.drop("iid", axis=1)
DATAexp.insert(12,"pid",DATAexp["partideology"].map(map_dict))
DATAexp.insert(10,"iid",DATAexp["itemideology"].map(map_dict))

DATAexp.head()


# In[33]:


get_ipython().run_line_magic('Rpush', 'DATAexp')


# In[34]:


get_ipython().run_cell_magic('R', '', '\nM <- lmer(RdeltaB ~ PE*partideology + scaledRpreI + (1 | partnum) + (1 | itemnum), data = DATAexp)\nprint(summary(M))')


# - Run the linear mixed model fit in full R syntax

# In[35]:


get_ipython().run_cell_magic('R', '', '\nM <- lmer(RdeltaB ~ PE:iid:pid + (1 | partnum) + (1 | itemnum), data = DATAexp)\nprint(summary(M))\n\nrfs <- ranef(M)$partnum\nffs <- fixef(M)')


# In[36]:


get_ipython().run_cell_magic('R', '', '\nM <- lmer(RdeltaB ~ scaledRpreI + PE + (1 | partnum) + (1 | itemnum), data = DATAexp)\nprint(summary(M))\n\nrfs <- ranef(M)$partnum\nffs <- fixef(M)')


# - Pull data back to the Python side

# In[38]:


get_ipython().run_line_magic('Rpush', 'DATAexp')


# In[41]:


get_ipython().run_cell_magic('R', '', '\nM <- lmer(RdeltaB ~ PE*pid*iid + (1 | partnum) + (1 | itemnum), data = DATAexp)\n# M <- lmer(RdeltaB ~ PE + pid + PE:pid + PE:iid + PE:pid:iid + (1 | partnum) + (1 | itemnum), data = DATAexp)\nprint(summary(M))\n\nrfs_part <- ranef(M)$partnum\nrfs_item <- ranef(M)$itemnum\nffs <- fixef(M)')


# In[42]:


get_ipython().run_line_magic('Rpull', 'rfs_part')
get_ipython().run_line_magic('Rpull', 'rfs_item')
get_ipython().run_line_magic('Rpull', 'ffs')


# In[43]:


# Fixed effects can be used to show linear predictions under the mixed model
fixedeffects = [ffs[i] for i in range(len(ffs))]
# Random effects can be used to show linear data with the random effects already removed
randomeffects = [[rfs_part[0][i] for i in range(len(rfs_part[0]))], [rfs_item[0][i] for i in range(len(rfs_item[0]))]]
rm_re = np.array([randomeffects[0][i] + randomeffects[1][j] for i in range(len(randomeffects[0])) for j in range(len(randomeffects[1]))])
if "RdeltaB_corrected" in DATAexp:
    DATAexp = DATAexp.drop("RdeltaB_corrected", axis=1)
DATAexp.insert(1,"RdeltaB_corrected",DATAexp["RdeltaB"] - rm_re)
DATAexp.head()


# In[44]:


# Declare groups over which to split here:
grp1 = "partideology"
grp2 = "itemideology"

fig, ax = plt.subplots(len(DATA[grp1].unique()), len(DATA[grp2].unique()), figsize=(8,5), sharey=True)
colors = ["royalblue", "crimson", "goldenrod"]

# Split dataframe into groups, where grpDATA is a subsampled dataframe of DATA with one split's data at a time
for name, grpDATA in DATAexp.groupby([grp1, grp2]):
    name = np.array(name, dtype=int) - 1
    cax = ax[name[0],name[1]]
    # Average the split's items within participants
    AVDATA = grpDATA.groupby(["partnum", "PE"], as_index=False).mean()
    
    # Plotting  x_estimator=np.mean
#     sns.regplot(x="PE", y="RdeltaB", scatter=True, data=AVDATA, \
#                 color=colors[name[-1]], ci=99, x_jitter=.1, marker=".", scatter_kws={'s':2}, \
#                 ax=cax)
    sns.regplot(x="PE", y="RdeltaB_corrected", x_estimator=np.mean, data=AVDATA,                 color=colors[name[-1]], ci=95, ax=cax)
    sns.despine()
    if name[0] == len(DATA[grp1].unique())-1:
        cax.set_xlabel("Prediction Error");
    else:
        cax.set_xticklabels("");
        cax.set_xlabel("");
    if name[1] == 0:
        cax.set_ylabel("Rational belief update");
    else:
        cax.set_ylabel("");
    
plt.tight_layout()
#plt.savefig('/Users/mada/Desktop/fig3rep.tif', dpi=300, format="tiff")

