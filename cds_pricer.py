import numpy as np
import datetime as datetime
from dateutil.relativedelta import *
from scipy.optimize import fsolve
import warnings
import copy


def month_year_indicator(tenor_strings):
	month_year_indicator = np.zeros(len(tenor_strings))
	for i in range(len(tenor_strings)):
		if tenor_strings[i][-1] == 'M':
			month_year_indicator[i] = 0
		else:
			month_year_indicator[i] = 1
	return month_year_indicator

def tenor_strings_to_dates(cob, tenor_strings):
	tenor_dates = np.empty(len(tenor_strings), dtype = datetime.date)
	for i in range(len(tenor_strings)):
		if tenor_strings[i][-1] == 'M':
			tenor_dates[i] = cob + relativedelta(months = int(tenor_strings[i][:-1]))
		else:
			tenor_dates[i] = cob + relativedelta(months = 12 * int(tenor_strings[i][:-1]))
	return tenor_dates


def day_count_30_360(start_date, end_date):
    """Returns number of days between start_date and end_date, using Thirty/360 convention"""
    d1 = min(30, start_date.day)
    d2 = min(d1, end_date.day) if d1 == 30 else end_date.day
    
    return 360*(end_date.year - start_date.year)\
           + 30*(end_date.month - start_date.month)\
           + d2 - d1


class imm:
	def __init__(self, cob, ir_curve):
		self.cob = cob
		self.ir_curve = ir_curve
		lower_bound_candidate = np.array([datetime.date(self.cob.year -1, 12, 20), datetime.date(self.cob.year, 3, 20), datetime.date(self.cob.year, 6, 20), datetime.date(self.cob.year, 9, 20), datetime.date(self.cob.year, 12, 20)])
		self.prev_imm_date = lower_bound_candidate[[x < self.cob for x in lower_bound_candidate]][-1]
		self.imm_dates = np.array([self.prev_imm_date  + relativedelta(months = 3 * i) for i in range(50)])
		self.imm_dates[0] = self.cob
		self.imm_dates_dfs = self.get_imm_dates_dfs()

	def get_imm_dates_dfs(self):
		imm_dates_dfs = np.zeros(len(self.imm_dates))
		imm_dates_dfs[0] = 1
		__, zeros = self.ir_curve.boostrap_dfs_zeros()
		for i in np.arange(1, len(self.imm_dates)):
			imm_dates_dfs[i] = self.ir_curve.interpolate_df(zeros, self.ir_curve.discount_method, self.imm_dates[i])
		return imm_dates_dfs

class discount_curve:
	def __init__(self, cob, tenor_strings, rates, discount_method):
		self.cob = cob
		self.discount_method = discount_method
		self.tenor_strings = tenor_strings
		self.rates	= np.array(rates)
		self.tenor_dates = tenor_strings_to_dates(self.cob, self.tenor_strings)

	def boostrap_dfs_zeros(self):
		dfs = np.zeros(len(self.tenor_strings))
		zeros = np.zeros(len(self.tenor_strings))
		a = month_year_indicator(self.tenor_strings)
		if self.discount_method == "ACT/360":
			for i in range(len(self.tenor_strings)):
				delta = self.tenor_dates[i] - self.cob
				if a[i] == 0:
					dfs[i] =  1 / (1 + self.rates[i] * delta.days / 360)
				else:
					dfs[i] = (1 - np.dot(dfs[:i] * self.rates[i], a[:i])) / (1 + self.rates[i])
				zeros[i] = dfs[i]**(-360/delta.days) - 1
		elif self.discount_method == "30/360":
			for i in range(len(self.tenor_strings)):
				delta = day_count_30_360(self.cob, self.tenor_dates[i])
				if a[i] == 0:
					dfs[i] =  1 / (1 + self.rates[i] * delta / 360)
				else:
					dfs[i] = (1 - np.dot(dfs[:i] * self.rates[i], a[:i])) / (1 + self.rates[i])
				zeros[i] = dfs[i]**(-360/delta) - 1
		else:
			print("To Do")
		return dfs, zeros

	def interpolate_df(self, zeros, discount_method, date):
		zeros = np.insert(zeros, 0, zeros[0])
		zeros = np.insert(zeros, -1, zeros[-1])
		dates = np.insert(self.tenor_dates, 0, self.cob)
		dates = np.insert(dates, -1, datetime.date(2099, 12, 31))
		diffs = dates - date
		for i in range(len(self.tenor_dates)):
			if (diffs[i].days <=0 and diffs[i+1].days > 0):
				c = i
			else:
				pass
		if self.discount_method == "ACT/360":
			delta_1 = date - dates[c]
			m = (zeros[c+1] - zeros[c]) / (dates[c+1] - dates[c]).days
			interp_zero = m * delta_1.days + zeros[c]
			delta_2 = date - self.cob
			interp_df = 1 / ((1+interp_zero)**(delta_2.days/360))
			return interp_df
		if self.discount_method == "30/360":
			delta_1 = day_count_30_360(dates[c], date)
			m = (zeros[c+1] - zeros[c]) / (dates[c+1] - dates[c]).days
			interp_zero = m * delta_1 + zeros[c]
			delta_2 = day_count_30_360(self.cob, date)
			interp_df = 1 / ((1+interp_zero)**(delta_2/360))
			return interp_df
		else:
			print("To Do")


class cds:
	def __init__(self, credit_spreads, recovery, maturities, cob, imm, ir_curve):
		self.maturities = maturities
		self.credit_spreads = credit_spreads
		self.recovery = recovery
		self.survival_probs = None
		self.default_probs = None
		self.hazard_rates = None
		self.cob = cob
		self.imm = imm
		self.ir_curve = ir_curve
		self.boostsrap_hazard_rates()

	def cds_definition(self, maturity, coupon, notional = 1e6):
		self.maturity = maturity
		self.coupon = coupon
		self.notional = notional

	def C_n1(self, imm_dates_dfs, imm_dates, credit_spread, h):
		func = 0
		for i in np.arange(1, len(imm_dates)):
			alpha = (imm_dates[i] - imm_dates[i-1]).days / 360
			delta_i = (imm_dates[i] - imm_dates[0]).days / 365
			delta_in1 = (imm_dates[i-1] - imm_dates[0]).days / 365
			func = func + delta_i * np.exp(-h * delta_in1) * (1 - self.recovery - credit_spread * alpha / 2 - np.exp(-h * (delta_i - delta_in1)) * (1 - self.recovery + credit_spread * alpha / 2))
		return func



	def C_nj(self, imm_dates_dfs, imm_dates, credit_spread, h, c2, const):
		func = - const
		for i in np.arange(c2, len(imm_dates)):
			alpha = (imm_dates[i] - imm_dates[i-1]).days / 360
			delta_i_in1 = (imm_dates[i] - imm_dates[i-1]).days / 365
			delta_in1_c2n1 = (imm_dates[i-1] - imm_dates[c2 - 1]).days / 365
			Aji = (1- self.recovery - credit_spread * alpha / 2) / (1- self.recovery + credit_spread * alpha / 2)
			func = func + imm_dates_dfs[i] * (1-self.recovery + credit_spread * alpha / 2)  * np.exp(-h * delta_in1_c2n1) * (Aji  - np.exp(-h * delta_i_in1))
		return func 


	def boostsrap_hazard_rates(self):
		survival_probs = [1]
		default_probs = [0]
		hazard_rates = []
		c1 = 0
		for i in range(len(self.imm.imm_dates)):
			if (self.maturities[0] - self.imm.imm_dates[i]).days >= 0:
				c1 += 1
			else:
				pass
		imm_dates_sliced = self.imm.imm_dates[:c1]
		imm_dates_dfs_sliced = self.imm.imm_dates_dfs[:c1]
		credit_spread = self.credit_spreads[0]

		func = lambda h: self.C_n1(imm_dates_dfs_sliced, imm_dates_sliced, credit_spread, h)
		initial_guess = 0
		solution = fsolve(func, initial_guess)
		hazard_rates.append(solution[0])

		for i in np.arange(1, len(imm_dates_sliced)):
			delta_i = (self.imm.imm_dates[i] - self.imm.imm_dates[i-1]).days / 365
			p = (np.exp(-solution * delta_i) * survival_probs[-1])[0]
			survival_probs.append(p)
			default_probs.append(1-p)

		for j in np.arange(1, len(self.maturities)):
			c1 = 0
			for i in range(len(self.imm.imm_dates)):
				if (self.maturities[j] - self.imm.imm_dates[i]).days >= 0:
					c1 += 1
				else:
					pass
			imm_dates_sliced = self.imm.imm_dates[:c1]
			imm_dates_dfs_sliced = self.imm.imm_dates_dfs[:c1]
			credit_spread = self.credit_spreads[j]

			c2 = 0
			for i in range(len(self.imm.imm_dates)):
				if (self.maturities[j-1] - self.imm.imm_dates[i]).days >= 0:
					c2 += 1
				else:
					pass 

			const = 0
			for i in np.arange(1, c2):
				alpha = (self.imm.imm_dates[i] - self.imm.imm_dates[i-1]).days / 360
				Aji = (1- self.recovery - credit_spread * alpha / 2) / (1- self.recovery + credit_spread * alpha / 2)
				const = const + imm_dates_dfs_sliced[i] * (self.recovery - 1 - credit_spread * alpha / 2) * (Aji * survival_probs[i-1] - survival_probs[i])

			const = const / survival_probs[c2-1]


			func = lambda h: self.C_nj(imm_dates_dfs_sliced, imm_dates_sliced, credit_spread, h, c2, const)

			initial_guess = 0
			solution = fsolve(func, initial_guess)
			hazard_rates.append(solution[0])


			for i in np.arange(c2, len(imm_dates_sliced)):
				delta_i = (self.imm.imm_dates[i] - self.imm.imm_dates[i-1]).days / 365
				p = (np.exp(-solution * delta_i) * survival_probs[-1])[0]
				survival_probs.append(p)
				default_probs.append(1-p)


		self.survival_probs = survival_probs
		self.default_probs = default_probs
		self.hazard_rates = hazard_rates
		return 

	def cds_premium_pv(self):
		c1 = 0
		for i in range(len(self.imm.imm_dates)):
			if (self.maturity - self.imm.imm_dates[i]).days >= 0:
				c1 += 1
			else:
				pass
		imm_dates_sliced = self.imm.imm_dates[:c1]
		imm_dates_dfs_sliced = self.imm.imm_dates_dfs[:c1]

		pv = 0 
		for i in np.arange(1, len(imm_dates_sliced)):
			alpha = (imm_dates_sliced[i] - imm_dates_sliced[i-1]).days / 360
			pv = pv + alpha * self.coupon * self.notional * self.survival_probs[i] * imm_dates_dfs_sliced[i]  + self.coupon / 2 * imm_dates_dfs_sliced[i] * alpha * (-self.survival_probs[i] + self.survival_probs[i-1]) * self.notional
		return pv

	def cds_default_pv(self):
		c1 = 0
		for i in range(len(self.imm.imm_dates)):
			if (self.maturity - self.imm.imm_dates[i]).days >= 0:
				c1 += 1
			else:
				pass
		imm_dates_sliced = self.imm.imm_dates[:c1]
		imm_dates_dfs_sliced = self.imm.imm_dates_dfs[:c1]
		pv = 0
		for i in np.arange(1, len(imm_dates_sliced)):
			pv = pv - (1 - self.recovery) * self.notional * (-self.survival_probs[i] + self.survival_probs[i-1]) * imm_dates_dfs_sliced[i]
		return pv


	def cds_clean_pv(self):
		return self.cds_premium_pv() + self.cds_default_pv()

	def cds_acrrued(self):
		alpha = (self.cob - self.imm.prev_imm_date).days / 360 - 1/360
		return alpha * self.coupon * self.notional

	def cds_dirty_pv(self):
		return self.cds_clean_pv() + self.cds_acrrued()









if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	cob = datetime.date(2022, 4, 29) 
	
	tenor_strings = ['1M', '2M', '3M', '6M', '9M', '1Y', '2Y','3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y']
	rates = np.array([0.8165, 1.0158, 1.1880, 1.6668, 2.0427, 2.3107, 2.7840, 2.8380, 2.8115, 2.7785, 2.7605, 2.7498, 2.7433, 2.7398, 2.7385]) / 100
	sofr_curve = discount_curve(cob, tenor_strings, rates, "ACT/360")
	imm_cob = imm(cob, sofr_curve)

	maturities = np.array([datetime.date(2022, 12, 20), datetime.date(2023, 6, 20), datetime.date(2024, 6, 20), datetime.date(2025, 6, 20), datetime.date(2026, 6, 20), datetime.date(2027, 6, 20), datetime.date(2029, 6, 20), datetime.date(2032, 6, 20)])
	credit_spreads = np.array([161, 198, 349, 476, 578, 665, 710, 730]) / 10000
	recovery = 0.4

	credit_spreads_default = credit_spreads + 1e6
	credit_spreads_p1bp = credit_spreads + 1/ 10000





	name_1 = cds(credit_spreads, recovery, maturities, cob, imm_cob, sofr_curve)
	
	print(name_1.hazard_rates)
	
	cds1 = copy.copy(name_1)
	maturity = datetime.date(2025, 6, 20)
	coupon = 500 / 10000
	notional = 8.5e6
	cds1.cds_definition(maturity, coupon, notional)
	
	fee_leg_pv = cds1.cds_premium_pv()
	print(f"fee leg PV of cds: {round(fee_leg_pv,2):,}")
	default_leg_pv = cds1.cds_default_pv()
	print(f"default leg PV of cds: {round(default_leg_pv,2):,}")
	clean_pv = cds1.cds_clean_pv()
	print(f"clean PV of cds: {round(clean_pv,2):,}")
	acrrued_pv = cds1.cds_acrrued()
	print(f"accrued PV of cds: {round(acrrued_pv,2):,}")
	dirty_pv = cds1.cds_dirty_pv()
	print(f"dirty PV of cds: {round(dirty_pv,2):,}")


	# cs01
	name_1_p1bp = cds(credit_spreads_p1bp, recovery, maturities, cob, imm_cob, sofr_curve)
	cds_1_p1bp = copy.copy(name_1_p1bp)
	cds_1_p1bp.cds_definition(maturity, coupon, notional)
	cs01 = cds_1_p1bp.cds_clean_pv() - clean_pv
	print(f"cs01 of cds: {round(cs01,2):,}")


	name_1_default = cds(credit_spreads_default, recovery, maturities, cob, imm_cob, sofr_curve)
	cds_1_default = copy.copy(name_1_default)
	cds_1_default.cds_definition(maturity, coupon, notional)	
	jtd = cds_1_default.cds_clean_pv() - clean_pv






















	