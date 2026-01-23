"""
Dataset-specific configuration for multi-dataset pipeline.
"""

DATASET_CONFIG = {
    '01_creditcard.csv': {
        'name': 'Credit Card Fraud',
        'type': 'numeric_temporal',
        'label_col': 'Class',
        'drop_cols': ['Time'],  # Temporal column - handle separately
        'temporal_col': 'Time',
        'requires_encoding': False,
        'numeric_cols': None,  # Auto-detect all except label
        'categorical_cols': [],
        'test_size': 0.4,
        'stratify': True,
        'shuffle': False,  # Time-aware split
    },
    '03_fraud_oracle.csv': {
        'name': 'Vehicle Claim Fraud',
        'type': 'mixed',
        'label_col': 'FraudFound_P',
        'drop_cols': [],
        'temporal_col': None,
        'requires_encoding': True,
        'categorical_cols': [
            'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed',
            'MonthClaimed', 'Sex', 'MaritalStatus', 'Fault', 'PolicyType',
            'VehicleCategory', 'VehiclePrice', 'Days_Policy_Accident',
            'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle',
            'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
            'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim',
            'NumberOfCars', 'BasePolicy'
        ],
        'test_size': 0.4,
        'stratify': True,
        'shuffle': False,
    },
    '04_bank_account.csv': {
        'name': 'Bank Account Fraud',
        'type': 'mixed',
        'label_col': 'fraud_bool',
        'drop_cols': [],
        'temporal_col': 'month',  # Optional temporal column
        'requires_encoding': True,
        'categorical_cols': [
            'payment_type', 'employment_status', 'housing_status', 'source', 'device_os'
        ],
        'test_size': 0.4,
        'stratify': True,
        'shuffle': True,
    },
    '05_online_payment.csv': {
        'name': 'Online Payment Fraud',
        'type': 'mixed',
        'label_col': 'isFraud',
        'drop_cols': ['nameOrig', 'nameDest'],  # Drop PII columns
        'temporal_col': 'step',
        'requires_encoding': True,
        'categorical_cols': ['type'],
        'test_size': 0.4,
        'stratify': True,
        'shuffle': False,  # Keep temporal order
    },
}