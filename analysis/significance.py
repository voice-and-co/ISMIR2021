import os
import json
import pandas as pd
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


def anova_table(aov):
    # ANOVA test
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov


def kruskal_2samp(df,selected_grades,test=stats.ks_2samp):
    # Kruskal test
    df_temp = df[df['grade'].isin(selected_grades)]
    # non-parametric tests
    selected_data = [df['mean'][df['grade'] == g] for g in selected_grades]
    stat, p = stats.kruskal(selected_data[0],selected_data[1])
    print('Kruskal for grades: {},{} Statistics={:.3f}, p={:.3f}'.format(int(selected_grades[0]),int(selected_grades[1]),stat, p))

def ks_2samp(df,selected_grades):
    # Kolgomorov-Smirnov Analysis
    df_temp = df[df['grade'].isin(selected_grades)]
    # non-parametric tests
    selected_data = [df['mean'][df['grade'] == g] for g in selected_grades]
    stat, p = stats.ks_2samp(selected_data[0],selected_data[1],'less', mode='exact')
    print('KS for grades: {},{} Statistics={:.3f}, p={:.3f}'.format(int(selected_grades[0]),int(selected_grades[1]),stat, p))


def all_2samp(df,selected_grades,test=stats.ks_2samp):
    # Comparing different tests
    df_temp = df[df['grade'].isin(selected_grades)]
    # non-parametric tests
    selected_data = [df['mean'][df['grade'] == g] for g in selected_grades]
    stat1, p1 = stats.kruskal(selected_data[0],selected_data[1])
    stat2, p2 = stats.ks_2samp(selected_data[0],selected_data[1],'less', mode='exact')
    stat3, p3 = stats.ks_2samp(selected_data[0],selected_data[1],mode='exact')
    print('Grades: {},{} Stats_KW={:.3f}, p_KW={:.3f}, Stats_KS_less={:.3f}, p_KS_less={:.3f}, Stats_KS={:.3f}, p_KS={:.3f}'.format(int(selected_grades[0]),int(selected_grades[1]),stat1, p1, stat2, p2, stat3, p3))


feat_path = '../features/'
for collection in ['classical', 'modern']:
    for feature in ['contour', 'melody_pattern_coincidence', 'onset', 'rhythm_pattern_coincidence', 'harmony']:
        filename = feature + '_' + collection + '.json'
        with open(os.path.join(feat_path,filename)) as json_file:
            print(filename)
            data = json.load(json_file)
            scores = [[v['grade'],v['statistics']['mean']] for k,v in data.items()]
            df = pd.DataFrame(scores,columns=['grade','mean'])

            #### comparison between so many grades always shows as significant because of a few pairs, so this is not very informative
            # selected_grades=[str(i) for i in range(9)]
            # #stats
            # df_temp =  df[df['grade'].isin(selected_grades)]
            # ### non-parametric tests
            # selected_data = [df['mean'][df['grade'] == g] for g in selected_grades]
            # stat, p = stats.kruskal(selected_data[0],selected_data[1],selected_data[2],selected_data[3],selected_data[4],selected_data[5],selected_data[6],selected_data[7],selected_data[8])
            # print('Statistics=%.3f, p=%.3f' % (stat, p))


            for i in range(8):
                for j in range(1,8):
                    if (i+j)<9:
                        selected_grades=[str(i),str(i+j)]
                        # kruskal_2samp(df,selected_grades)
                        # ks_2samp(df,selected_grades)
                        all_2samp(df,selected_grades)

            #import pdb;pdb.set_trace()



            # ##### we can't apply parametric tests because assumptions are violated
            # # model = ols('mean ~ C(grade)', data=df_temp).fit()
            # # aov_table = sm.stats.anova_lm(model, typ=2)
            # # print(anova_table(aov_table))

            # ### check the assumptions

            # ### for normality the p should be high (non-significance)
            # print(stats.shapiro(model.resid))
            # # ##optional plot the residual
            # # fig = plt.figure(figsize= (10, 10))
            # # ax = fig.add_subplot(111)
            # # normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
            # # ax.set_title("Probability plot of model residual's", fontsize= 20)
            # # ax.set
            # # plt.show()

            # ### check variance
            # selected_data = [df['mean'][df['grade'] == g] for g in selected_grades]
            # print(stats.levene(selected_data[0],selected_data[1]))
            # # ##optional plot of variance, p should be high
            # # fig = plt.figure(figsize= (10, 10))
            # # ax = fig.add_subplot(111)
            # # ax.set_title("Box Plot", fontsize= 20)
            # # ax.set
            # # ax.boxplot(selected_data,
            # #            labels= selected_grades,
            # #            showmeans= True)
            # # plt.xlabel("Drug Dosage")
            # # plt.ylabel("Libido Score")
            # # plt.show()




