# Data Documentation

## Overview
This directory contains all data files used in the Lending Club Sentiment Analysis project.

## Data Sources

### Primary Dataset: Lending Club
- **Source**: Kaggle - Lending Club Dataset
- **URL**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Size**: ~2.26M records (sampled to 100K for analysis)
- **License**: Public domain
- **Citation**: Lending Club (2019). Lending Club Loan Data. Kaggle.

### Download Instructions

#### Option 1: Kaggle API (Recommended)
```bash
# Install kaggle CLI
pip install kaggle

# Configure your Kaggle API credentials
# Download from: https://www.kaggle.com/settings/account
# Place kaggle.json in ~/.kaggle/

# Download the dataset
kaggle datasets download -d wordsforthewise/lending-club

# Extract to data/raw/
unzip lending-club.zip -d data/raw/
```

#### Option 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Click "Download" button
3. Extract the ZIP file to `data/raw/`

#### Option 3: Direct Link
- Direct download: https://www.kaggle.com/download/wordsforthewise/lending-club

## Data Structure

### Raw Data (`data/raw/`)
- `lending_club_data.csv`: Original Lending Club dataset
- `metadata.json`: Dataset metadata and schema

### Processed Data (`data/processed/`)
- `cleaned_data.csv`: Cleaned and preprocessed dataset
- `synthetic_descriptions.csv`: Generated loan descriptions
- `feature_engineered.csv`: Final dataset with all features

### External Data (`data/external/`)
- Additional datasets used for validation or comparison

## Data Schema

### Key Variables
- `loan_amnt`: Loan amount requested
- `purpose`: Purpose of the loan
- `grade`: Lending Club assigned grade
- `emp_length`: Employment length
- `annual_inc`: Annual income
- `dti`: Debt-to-income ratio
- `delinq_2yrs`: Number of 30+ days past-due incidences
- `inq_last_6mths`: Number of inquiries in past 6 months
- `open_acc`: Number of open credit lines
- `pub_rec`: Number of derogatory public records
- `revol_bal`: Total credit revolving balance
- `revol_util`: Revolving line utilization rate
- `total_acc`: Total number of credit lines
- `initial_list_status`: Initial listing status
- `out_prncp`: Remaining outstanding principal
- `out_prncp_inv`: Remaining outstanding principal for portion of total amount funded by investors
- `total_pymnt`: Payments received to date for total amount funded
- `total_pymnt_inv`: Payments received to date for portion of total amount funded by investors
- `total_rec_prncp`: Principal received to date
- `total_rec_int`: Interest received to date
- `total_rec_late_fee`: Late fees received to date
- `recoveries`: Post charge off gross recovery
- `collection_recovery_fee`: Post charge off collection fee
- `last_pymnt_d`: Last month payment was received
- `last_pymnt_amnt`: Last total payment amount received
- `next_pymnt_d`: Next scheduled payment date
- `last_credit_pull_d`: The most recent month LC pulled credit for this loan
- `last_fico_range_high`: The upper boundary range the borrower's last FICO at the time of credit pull
- `last_fico_range_low`: The lower boundary range the borrower's last FICO at the time of credit pull
- `collections_12_mths_ex_med`: Number of collections in 12 months excluding medical collections
- `mths_since_last_major_derog`: Months since most recent 90-day or worse rating
- `policy_code`: Publicly available policy_code=1, new products not publicly available policy_code=2
- `application_type`: Indicates whether the loan is an individual application or a joint application with two co-borrowers
- `annual_inc_joint`: The combined self-reported annual income provided by the co-borrowers during registration
- `dti_joint`: A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income
- `verification_status_joint`: Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
- `acc_now_delinq`: The number of accounts on which the borrower is now delinquent
- `tot_coll_amt`: Total collection amounts ever owed
- `tot_cur_bal`: Total current balance of all accounts
- `open_acc_6m`: Number of open trades in last 6 months
- `open_il_6m`: Number of currently active installment trades
- `open_il_12m`: Number of installment accounts opened in past 12 months
- `open_il_24m`: Number of installment accounts opened in past 24 months
- `mths_since_rcnt_il`: Months since most recent installment accounts opened
- `total_bal_il`: Total current balance of all installment accounts
- `il_util`: Ratio of total current balance to high credit/credit limit on all install acct
- `open_rv_12m`: Number of revolving trades opened in past 12 months
- `open_rv_24m`: Number of revolving trades opened in past 24 months
- `max_bal_bc`: Maximum current balance owed on all revolving accounts
- `all_util`: Balance to credit limit on all trades
- `total_rev_hi_lim`: Total revolving high credit/credit limit
- `inq_fi`: Number of personal finance inquiries
- `total_cu_tl`: Number of finance trades
- `inq_last_12m`: Number of credit inquiries in past 12 months
- `acc_open_past_24mths`: Number of trades opened in past 24 months
- `avg_cur_bal`: Average current balance of all accounts
- `bc_open_to_buy`: Total open to buy on revolving bankcards
- `bc_util`: Ratio of total current balance to high credit/credit limit for revolving bankcards
- `chargeoff_within_12_mths`: Number of charge-offs within 12 months
- `delinq_amnt`: Past due amount owed for the accounts on which the borrower is now delinquent
- `mo_sin_old_il_acct`: Months since oldest bank installment account opened
- `mo_sin_old_rev_tl_op`: Months since oldest revolving account opened
- `mo_sin_rcnt_rev_tl_op`: Months since most recent revolving account opened
- `mo_sin_rcnt_tl`: Months since most recent account opened
- `mort_acc`: Number of mortgage accounts
- `mths_since_recent_bc`: Months since most recent bankcard account opened
- `mths_since_recent_bc_dlq`: Months since most recent bankcard delinquency
- `mths_since_recent_inq`: Months since most recent inquiry
- `mths_since_recent_revol_delinq`: Months since most recent revolving delinquency
- `num_accts_ever_120_pd`: Number of accounts ever 120 or more days past due
- `num_actv_bc_tl`: Number of currently active bankcard accounts
- `num_actv_rev_tl`: Number of currently active revolving trades
- `num_bc_sats`: Number of satisfactory bankcard accounts
- `num_bc_tl`: Number of bankcard accounts
- `num_il_tl`: Number of installment accounts
- `num_op_rev_tl`: Number of open revolving accounts
- `num_rev_accts`: Number of revolving accounts
- `num_rev_tl_bal_gt_0`: Number of revolving trades with balance >0
- `num_sats`: Number of satisfactory accounts
- `num_tl_120dpd_2m`: Number of accounts 120 days past due (updated in past 2 months)
- `num_tl_30dpd`: Number of accounts 30 days past due (updated in past 2 months)
- `num_tl_90g_dpd_24m`: Number of accounts 90 or more days past due in last 24 months
- `num_tl_op_past_12m`: Number of accounts opened in past 12 months
- `pct_tl_nvr_dlq`: Percent of trades never delinquent
- `percent_bc_gt_75`: Percentage of all bankcard accounts > 75% of limit
- `pub_rec_bankruptcies`: Number of public record bankruptcies
- `tax_liens`: Number of tax liens
- `tot_hi_cred_lim`: Total high credit/credit limit
- `total_bal_ex_mort`: Total balance excluding mortgage
- `total_bc_limit`: Total bankcard high credit/credit limit
- `total_il_high_credit_limit`: Total installment high credit/credit limit

### Target Variables
- `target_5%`: Binary target for 5% default rate regime
- `target_10%`: Binary target for 10% default rate regime  
- `target_15%`: Binary target for 15% default rate regime

## Data Quality

### Missing Values
- Text descriptions: ~93% missing (synthetic generation used)
- Employment length: ~5% missing
- Annual income: ~2% missing
- Other variables: <1% missing

### Data Cleaning Steps
1. Remove duplicate records
2. Handle missing values (imputation/removal)
3. Generate synthetic text descriptions
4. Create target variables based on default status
5. Feature engineering and scaling

## Privacy and Ethics
- All personal identifiers have been removed
- Data is anonymized and aggregated
- Used for academic research purposes only
- Complies with data protection regulations

## Citation
When using this data, please cite:
```
Lending Club (2019). Lending Club Loan Data. Kaggle.
https://www.kaggle.com/datasets/wordsforthewise/lending-club
```
