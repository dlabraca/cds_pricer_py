import numpy as np
import datetime as datetime
from dateutil.relativedelta import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import warnings


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

class discount_curve:
		def __init__(self, cob, tenors_strings, rates, discount_method):
			self.cob = cob
			self.discount_method = discount_method
			self.tenor_strings = tenor_strings
			self.rates	= np.array(rates) / 100
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
				m = (zeros[c+1] - zeros[c-1]) / (dates[c+1] - dates[c-1]).days
				interp_zero = m * delta_1.days + zeros[c]
				delta_2 = date - self.cob
				interp_df = 1 / ((1+interp_zero)**(delta_2.days/360))
				return interp_df
			if self.discount_method == "30/360":
				delta_1 = day_count_30_360(dates[c], date)
				m = (zeros[c+1] - zeros[c-1]) / (dates[c+1] - dates[c-1]).days
				interp_zero = m * delta_1 + zeros[c]
				delta_2 = day_count_30_360(self.cob, date)
				interp_df = 1 / ((1+interp_zero)**(delta_2/360))
				return interp_df
			else:
				print("To Do")


class cds_mkt_data:
	def __init__(self, credit_spreads, recovery, maturities, cob):
		self.maturities = maturities
		self.credit_spreads = credit_spreads
		self.recovery = recovery
		self.survival_probs = None
		self.survival_probs = None
		self.cob = cob
		self.imm_dates = None
		self.imm_dates_dfs = None

	def get_imm_dates_lower_bound(self):
		lower_bound_candidate = np.array([datetime.date(self.cob.year -1, 12, 20), datetime.date(self.cob.year, 3, 20), datetime.date(self.cob.year, 6, 20), datetime.date(self.cob.year, 9, 20), datetime.date(self.cob.year, 12, 20)])
		return lower_bound_candidate[[x < self.cob for x in lower_bound_candidate]][-1]

	def get_imm_dates(self):
		imm_dates = np.array([self.get_imm_dates_lower_bound()  + relativedelta(months = 3 * i) for i in range(50)])
		imm_dates[0] = self.cob
		self.imm_dates = imm_dates
		return imm_dates

	def get_imm_dates_dfs(self, sofr_curve):
		imm_dates = self.get_imm_dates()
		imm_dates_dfs = np.zeros(len(imm_dates))
		imm_dates_dfs[0] = 1
		__, zeros = sofr_curve.boostrap_dfs_zeros()
		for i in np.arange(1, len(imm_dates)):
			imm_dates_dfs[i] = sofr_curve.interpolate_df(zeros, sofr_curve.discount_method, imm_dates[i])
		self.imm_dates_dfs = imm_dates_dfs
		return imm_dates_dfs



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


	def boostsrap_hazard_rates(self, sofr_curve):
		survival_probs = [1]
		default_probs = [0]
		hazard_rates = []
		c1 = 0
		for i in range(len(self.imm_dates)):
			if (self.maturities[0] - self.imm_dates[i]).days >= 0:
				c1 += 1
			else:
				pass
		imm_dates_sliced = self.imm_dates[:c1]
		imm_dates_dfs_sliced = self.imm_dates_dfs[:c1]
		credit_spread = self.credit_spreads[0]
		func = lambda h: self.C_n1(imm_dates_dfs_sliced, imm_dates_sliced, credit_spread, h)
		initial_guess = 0
		solution = fsolve(func, initial_guess)
		hazard_rates.append(solution)

		for i in np.arange(1, len(imm_dates_sliced)):
			delta_i = (self.imm_dates[i] - self.imm_dates[i-1]).days / 365
			p = (np.exp(-solution * delta_i) * survival_probs[-1])[0]
			survival_probs.append(p)
			default_probs.append(1-p)

		for j in np.arange(1, len(maturities)):
			c1 = 0
			for i in range(len(self.imm_dates)):
				if (self.maturities[j] - self.imm_dates[i]).days >= 0:
					c1 += 1
				else:
					pass
			imm_dates_sliced = self.imm_dates[:c1]
			imm_dates_dfs_sliced = self.imm_dates_dfs[:c1]
			credit_spread = self.credit_spreads[j]

			c2 = 0
			for i in range(len(self.imm_dates)):
				if (self.maturities[j-1] - self.imm_dates[i]).days >= 0:
					c2 += 1
				else:
					pass

			const = 0
			for i in np.arange(1, c2):
				alpha = (self.imm_dates[i] - self.imm_dates[i-1]).days / 360
				Aji = (1- self.recovery - credit_spread * alpha / 2) / (1- self.recovery + credit_spread * alpha / 2)
				const = const + imm_dates_dfs_sliced[i] * (self.recovery - 1 - credit_spread * alpha / 2) * (Aji * survival_probs[i-1] - survival_probs[i])

			const = const / survival_probs[c2-1] 

			func = lambda h: self.C_nj(imm_dates_dfs_sliced, imm_dates_sliced, credit_spread, h, c2, const)
			initial_guess = 0
			solution = fsolve(func, initial_guess)
			hazard_rates.append(solution)


			for i in np.arange(c2, len(imm_dates_sliced)):
				delta_i = (self.imm_dates[i] - self.imm_dates[i-1]).days / 365
				p = (np.exp(-solution * delta_i) * survival_probs[-1])[0]
				survival_probs.append(p)
				default_probs.append(1-p)


		self.survival_probs = survival_probs
		self.default_probs = default_probs
		return survival_probs, default_probs

	def cds_premium_pv(self, maturity, coupon, notional):
		c1 = 0
		for i in range(len(self.imm_dates)):
			if (maturity - self.imm_dates[i]).days >= 0:
				c1 += 1
			else:
				pass
		imm_dates_sliced = self.imm_dates[:c1]
		imm_dates_dfs_sliced = self.imm_dates_dfs[:c1]

		pv = 0 
		for i in np.arange(1, len(imm_dates_sliced)):
			alpha = (imm_dates_sliced[i] - imm_dates_sliced[i-1]).days / 360
			pv = pv + alpha * coupon * notional * self.survival_probs[i] * imm_dates_dfs_sliced[i]  + coupon / 2 * imm_dates_dfs_sliced[i] * alpha * (-self.survival_probs[i] + self.survival_probs[i-1]) * notional
		pv = pv + self.cds_acrrued(coupon, notional)
		return pv

	def cds_default_pv(self, maturity, notional):
		c1 = 0
		for i in range(len(self.imm_dates)):
			if (maturity - self.imm_dates[i]).days >= 0:
				c1 += 1
			else:
				pass
		imm_dates_sliced = self.imm_dates[:c1]
		imm_dates_dfs_sliced = self.imm_dates_dfs[:c1]

		pv = - (1 - self.recovery) * notional * (-self.survival_probs[1] + self.survival_probs[0]) * imm_dates_dfs_sliced[1]
		for i in np.arange(2, len(imm_dates_sliced)):
			pv = pv - (1 - self.recovery) * notional * (-self.survival_probs[i] + self.survival_probs[i-1]) * imm_dates_dfs_sliced[i]
		return pv

	def cds_clean_pv(self, maturity, coupon, notional):
		return self.cds_premium_pv(maturity, coupon, notional) + self.cds_default_pv(maturity, notional) - self.cds_acrrued(coupon, notional)

	def cds_acrrued(self, coupon, notional):
		alpha = (self.cob - self.get_imm_dates_lower_bound()).days / 360 - 1/360
		return alpha * coupon * notional

	def cds_dirty_pv(self, maturity, coupon, notional):
		return self.cds_clean_pv(maturity, coupon, notional) + self.cds_acrrued(coupon, notional)









if __name__ == "__main__":
	warnings.filterwarnings("ignore")
	cob = datetime.date(2022, 4, 29) 
	
	tenor_strings = ['1M', '2M', '3M', '6M', '9M', '1Y', '2Y','3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y']
	rates = [0.8165, 1.0158, 1.1880, 1.6668, 2.0427, 2.3107, 2.7840, 2.8380, 2.8115, 2.7785, 2.7605, 2.7498, 2.7433, 2.7398, 2.7385]
	sofr_curve = discount_curve(cob, tenor_strings, rates, "ACT/360")
	# sofr_curve = discount_curve(cob, tenor_strings, rates, "30/360")

	maturities = np.array([datetime.date(2022, 12, 20), datetime.date(2023, 6, 20), datetime.date(2024, 6, 20), datetime.date(2025, 6, 20), datetime.date(2026, 6, 20), datetime.date(2027, 6, 20), datetime.date(2029, 6, 20), datetime.date(2032, 6, 20)])
	credit_spreads = np.array([161, 198, 349, 476, 578, 665, 710, 730]) / 10000
	recovery = 0.4

	cds1 = cds_mkt_data(credit_spreads, recovery, maturities, cob)
	cds1.get_imm_dates_dfs(sofr_curve)
	cds1.boostsrap_hazard_rates(sofr_curve)

	fee_leg_pv = cds1.cds_premium_pv(datetime.date(2025, 6, 20), 500 / 10000, 8.5e6)
	print(f"fee leg PV of cds: {round(fee_leg_pv,2):,}")
	default_leg_pv = cds1.cds_default_pv(datetime.date(2025, 6, 20), 8.5e6)
	print(f"default leg PV of cds: {round(default_leg_pv,2):,}")
	clean_pv = cds1.cds_clean_pv(datetime.date(2025, 6, 20), 500 / 10000, 8.5e6)
	print(f"clean PV of cds: {round(clean_pv,2):,}")
	acrrued_pv = cds1.cds_acrrued(500 / 10000, 8.5e6)
	print(f"accrued PV of cds: {round(acrrued_pv,2):,}")
	dirty_pv = cds1.cds_dirty_pv(datetime.date(2025, 6, 20), 500 / 10000, 8.5e6)
	print(f"dirty PV of cds: {round(dirty_pv,2):,}")


	# cs01
	credit_spreads_01 = credit_spreads + 1/ 10000
	cds1_01 = cds_mkt_data(credit_spreads_01, recovery, maturities, cob)
	cds1_01.get_imm_dates_dfs(sofr_curve)
	cds1_01.boostsrap_hazard_rates(sofr_curve)
	
	cs01 = cds1_01.cds_clean_pv(datetime.date(2025, 6, 20), 500 / 10000, 8.5e6) - clean_pv
	print(f"cs01 of cds: {round(cs01,2):,}")


	credit_spreads_jtd = credit_spreads + 1e6
	cds1_jtd = cds_mkt_data(credit_spreads_jtd, recovery, maturities, cob)
	cds1_jtd.get_imm_dates_dfs(sofr_curve)
	cds1_jtd.boostsrap_hazard_rates(sofr_curve)
	
	jtd = cds1_jtd.cds_clean_pv(datetime.date(2025, 6, 20), 500 / 10000, 8.5e6) - clean_pv
	print(f"jtd of cds: {round(jtd,1):,}")























	