from cds_pricer import discount_curve, imm, cds
import numpy as np
import datetime
from dateutil.relativedelta import *
from scipy.optimize import fsolve
import pandas as pd
import copy



class cds_index:
	def __init__(self, portfolio, maturities, cob, imm_curve, ir_curve):
		self.portfolio = portfolio
		self.N_portfolio = len(portfolio.index)
		self.maturities = [datetime.datetime.strptime(date_string, '%d-%b-%y').date() for date_string in np.array(maturities["date"])]
		self.cob = cob
		self.imm_curve = imm_curve
		self.ir_curve = ir_curve
		self.member_weights = np.array(self.portfolio['Weight'])
		self.member_recoveries = np.array(self.portfolio['Recovery'])

	def index_definition(self, maturity, coupon, notional, index_upf):
		self.maturity = maturity
		self.coupon = coupon
		self.notional = notional
		self.index_upf = index_upf
		self.scale = 1
		func = lambda h: self.calibrate_to_upf(h)
		solution = fsolve(func, self.scale, maxfev = 3)
		self.scale = solution[0]

	def create_portfolio_cds_objects_with_scale(self):
		portfolio_cds_objects_scaled = np.empty((self.N_portfolio, ), dtype = cds)
		for i in range(self.N_portfolio):
			credit_spreads_i = np.array([self.portfolio_scaled.iloc[i]['6M'], self.portfolio_scaled.iloc[i]['1Y'], self.portfolio_scaled.iloc[i]['2Y'], self.portfolio_scaled.iloc[i]['3Y'], self.portfolio_scaled.iloc[i]['4Y'], self.portfolio_scaled.iloc[i]['5Y'], self.portfolio_scaled.iloc[i]['7Y'], self.portfolio_scaled.iloc[i]['10Y']])
			recovery_i = self.portfolio.iloc[i]['Recovery']
			cds_object = cds(credit_spreads_i, recovery_i, self.maturities, self.cob, self.imm_curve, self.ir_curve)
			cds_object.cds_definition(self.maturity, self.coupon, self.notional)
			portfolio_cds_objects_scaled[i] = cds_object
		self.portfolio_cds_objects_scaled = portfolio_cds_objects_scaled
		return 

	def scale_portfolio(self, scale):
		self.portfolio_scaled = copy.copy(self.portfolio)
		self.scale = scale
		self.portfolio_scaled['6M'] = self.portfolio['6M'] * scale
		self.portfolio_scaled['1Y'] = self.portfolio['1Y'] * scale
		self.portfolio_scaled['2Y'] = self.portfolio['2Y'] * scale
		self.portfolio_scaled['3Y'] = self.portfolio['3Y'] * scale
		self.portfolio_scaled['4Y'] = self.portfolio['4Y'] * scale
		self.portfolio_scaled['5Y'] = self.portfolio['5Y'] * scale
		self.portfolio_scaled['7Y'] = self.portfolio['7Y'] * scale
		self.portfolio_scaled['10Y'] = self.portfolio['10Y'] * scale
		self.create_portfolio_cds_objects_with_scale()
		return

	def decomposed_index_value_upf(self):
		return np.dot([self.portfolio_cds_objects_scaled[i].cds_clean_pv() for i in range(self.N_portfolio)], self.member_weights) / self.notional




	def calibrate_to_upf(self, scale):
		self.scale_portfolio(scale)
		self.create_portfolio_cds_objects_with_scale()
		return self.decomposed_index_value_upf() + self.index_upf








if __name__ == "__main__":

	cob = datetime.date(2017, 5, 31) 

	ir_df = pd.read_csv('rates.csv')
	portfolio_df = pd.read_csv('portfolio.csv')
	maturities_df = pd.read_csv('maturities.csv')
	index_upf = pd.read_csv('index_marks.csv')['upf'].iloc[0]

	sofr_curve = discount_curve(cob, ir_df["Tenor"], ir_df["sofr"], "ACT/360")
	imm_curve = imm(cob, sofr_curve)

	index_1 = cds_index(portfolio_df, maturities_df, cob, imm_curve, sofr_curve)
	maturity = datetime.date(2021, 12, 20)
	coupon = 100 / 10000
	notional = 1e6
	index_1.index_definition(maturity, coupon, notional, index_upf)
	print(index_1.index_upf)
	print(index_1.scale)
	print(index_1.decomposed_index_value_upf())









