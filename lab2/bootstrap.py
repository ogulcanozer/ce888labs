import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np


def boostrap(sample, sample_size, iterations, ci):
        
        means = np.zeros(iterations)

        lw = (100-ci)/2 #Get the lower percentile from ci
        up = lw + ci #Get the upper percentile
        for iteration in range(0,iterations):
                new_sample = np.random.choice(sample, size=sample_size, replace=True)
                means[iteration] = np.mean(new_sample)
                
        data_mean = np.mean(means)
        lower = np.percentile(means, lw)
        upper = np.percentile(means, up)
        
        return data_mean, lower, upper


if __name__ == "__main__":
        
        #Boostrap Exercise 1
	df = pd.read_csv('./salaries.csv')
        
	data = df.values.T[1]
	boots = []
	for i in range(100, 100000, 1000):
		boot = boostrap(data, data.shape[0], i, 95 )
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])
		
	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")
	sns_plot.axes[0, 0].set_ylim(0,)
	sns_plot.axes[0, 0].set_xlim(0, 100000)
	sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')
        
	print(("Mean: %f")%(np.mean(data)))
	print(("Var: %f")%(np.var(data)))

	#Boostrap Exercise 2
	df = pd.read_csv('./vehicles.csv')

	#Current Fleet
	current = []
	c_data = df.values.T[0]
	print(c_data)
	current = boostrap(c_data, c_data.shape[0], 100000 , 95 )
	c_mean = current[0]
	c_lower = current[1]
	c_upper = current[2]

        #Proposed Fleet
	new = []
	n_data = df.values.T[1]
	n_data = n_data[np.logical_not(np.isnan(n_data))]
	print(n_data)
	new = boostrap(n_data, n_data.shape[0], 100000, 95 )
	n_mean = new[0]
	n_lower = new[1]
	n_upper = new[2]
	
	print("Upper difference :")
	print(n_upper-c_upper)
	print("Mean difference :")
	print(n_mean-c_mean)
	print("Lower difference :")
	print(n_lower-c_lower)

#End
