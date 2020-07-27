import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import runge_kutta_methods as RK
from scipy import interpolate
import scipy.optimize as opt
import argparse
import os

def parse_args():
	
	""" Calling the Training Statistics Data For Visualization """

	parser = argparse.ArgumentParser('Visualize your data statistics comparing multiple configurations')
	parser.add_argument('-c','--country' , type = str , help = 'Country to analize')
	parser.add_argument('-s','--susceptibles', type = int , help = "Number of susceptible people to get infected. Depends on each country's population size")
	parser.add_argument('-i','--infected' , type = int , help='Number of infected of day 1')
	parser.add_argument('-r','--recovered' , type=int , default=0,help='Number of recovered people on day 1 (defaults to 0)')
	parser.add_argument('-a','--a_estimate' , type=float , help = '"a" parameter first estimation')
	parser.add_argument('-b','--b_estimate' , type=float , help="'b' parameter first estimation")
	parser.add_argument('--days',type = int , help = 'Number of days to see the infected evolution')

	return parser.parse_args()


''' Make directory function '''
def make_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


# Idea es encontrar parámetros 

class SIR_Model:

	def __init__(self, Susceptibles , Infected  , Recovered , a_parameter , b_parameter , country , integration_time , dtStep = 0.01):
		
		# Initial Conditions: 
		self.S = Susceptibles
		self.I = Infected
		self.R = Recovered
		self.N = Susceptibles+Infected+Recovered

		# # SIR model estimated parameters
		self.a_parameter = a_parameter
		self.b_parameter = b_parameter

		self.dtStep = dtStep
		self.integration_time = integration_time

		# Country 
		self.country = country

	def get_data(self ,  country_name):
		df = pd.read_csv('time_series_covid19_confirmed_global.csv')
		countryDF = df[df['Country/Region'] == country_name] 
		country_df = countryDF.iloc[0].loc[Country_Start_Day[country_name]:]
		cases = country_df.values.tolist()
		cases.insert(0, 0)
		cases = np.array(cases)
		daily_cases = cases[1:] - cases[:-1]
		days = list(range(len(daily_cases)))
		self.contagion_days=len(days)
		return days, daily_cases

	def SIR_system(self,a,b):
		dI = lambda t,y1,y2,y3 : - a * y1 * y2
		dS = lambda t,y1,y2,y3 :   a * y1 * y2 - b * y2
		dR = lambda t,y1,y2,y3 :   b * y2
		return dI,dS,dR


	def solver(self,interpolated_time,a,b):

		# Model Initial Conditions
		vector0 = [self.S,self.I,self.R]
		t_init = 0
		time_max = self.contagion_days
		dtt = self.dtStep

		# SIR model differential equations
		f1,f2,f3 = self.SIR_system(a,b)

		# We use a Runge Kutta 4th order solver
		time, susceptibles , infected, recovered = RK.runge_kutta_4th_order_3_coupled(t_init,time_max,dtt,vector0,f1,f2,f3)

		# Spline interpolation to obtain data in discrete days 
		coeff_solver = interpolate.splrep(time, infected, s=0)
		Inf_Interp = interpolate.splev(interpolated_time, coeff_solver, der=0)

		return Inf_Interp

	def Plot_Results(self,day_array,cases_array,time_array,infect_array,x_title,y_title,aparameter,bparameter):
		fig, ax = plt.subplots(figsize = (8, 6))
		ax.set_xlabel(x_title, fontsize = 'large')
		ax.set_ylabel(y_title, fontsize = 'large')
		ax.plot(day_array, cases_array , linewidth = 1 )
		ax.plot(time_array,infect_array,linewidth=1,label = f'R0: {aparameter*self.N/bparameter}')
		ax.grid(True)
		plt.title(f'Covid-19 {self.country} Cases')
		plt.legend()
		name = f"{self.country}.png"
		fig.savefig(os.path.join(path,name))	
		plt.show()

	def FitCurve(self,solver_function,numdays,dailyCases,a_guess,b_guess):
		optimal_parameters, covariance = opt.curve_fit(solver_function, numdays, dailyCases , method = 'lm', p0 = [a_guess,b_guess])
		return optimal_parameters,covariance

	def run(self):
		days_data,daily_cases_data = self.get_data(self.country)
		opt,cov = self.FitCurve(self.solver,days_data,daily_cases_data,self.a_parameter,self.b_parameter)

		a_param = opt[0]
		b_param = opt[1]

		didt,dsdt,drdt = self.SIR_system(a_param,b_param)

		t0 = 0
		tmax = self.integration_time
		dt = self.dtStep
		N = self.N

		vec0 = [self.S,self.I,self.R]

		tRK,SusRK,InfRK,RecRK = RK.runge_kutta_4th_order_3_coupled(t0,tmax,dt,vec0,didt,dsdt,drdt)

		self.Plot_Results(days_data,daily_cases_data,tRK,InfRK,"Days","Infected",a_param,b_param)


if __name__ == '__main__':

	# Countries with best fitting model
	Country_Start_Day = {'Italy':'1/31/20','Germany':'1/27/20','India':'1/30/20','Spain':'1/31/20'}

	# Save Images:
	path = os.path.join(os.path.dirname(__file__), 'Results' )
	make_dir(path)

	# Intro Parameters to the Model
	args = parse_args()
	Country = args.country
	Susceptibles = args.susceptibles
	Infected = args.infected
	Recovered = args.recovered
	a_estimate = args.a_estimate
	b_estimate = args.b_estimate
	days_to_plot = args.days

	Model = SIR_Model(Susceptibles,Infected,Recovered,a_estimate,b_estimate, Country, days_to_plot ,0.01)
	Model.run()