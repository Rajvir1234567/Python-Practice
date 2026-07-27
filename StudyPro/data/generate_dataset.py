import pandas as pd
import numpy as np
import os

def generate_uci_student_dataset():
    np.random.seed(42)
    n_samples = 395

    schools = np.random.choice(['GP', 'MS'], size=n_samples, p=[0.88, 0.12])
    sexes = np.random.choice(['F', 'M'], size=n_samples, p=[0.53, 0.47])
    ages = np.random.randint(15, 22, size=n_samples)
    addresses = np.random.choice(['U', 'R'], size=n_samples, p=[0.78, 0.22])
    famsizes = np.random.choice(['LE3', 'GT3'], size=n_samples, p=[0.29, 0.71])
    pstatuses = np.random.choice(['T', 'A'], size=n_samples, p=[0.89, 0.11])
    
    medus = np.random.randint(0, 5, size=n_samples)
    fedus = np.random.randint(0, 5, size=n_samples)
    mjobs = np.random.choice(['at_home', 'health', 'other', 'services', 'teacher'], size=n_samples)
    fjobs = np.random.choice(['at_home', 'health', 'other', 'services', 'teacher'], size=n_samples)
    reasons = np.random.choice(['home', 'reputation', 'course', 'other'], size=n_samples)
    guardians = np.random.choice(['mother', 'father', 'other'], size=n_samples, p=[0.70, 0.24, 0.06])
    
    traveltimes = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.65, 0.27, 0.05, 0.03])
    studytimes = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.27, 0.50, 0.17, 0.06])
    failures = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.80, 0.13, 0.04, 0.03])
    
    binary_yes_no = lambda p: np.random.choice(['yes', 'no'], size=n_samples, p=[p, 1-p])
    schoolsups = binary_yes_no(0.13)
    famsups = binary_yes_no(0.61)
    paids = binary_yes_no(0.46)
    activities = binary_yes_no(0.51)
    nurseries = binary_yes_no(0.79)
    highers = binary_yes_no(0.95)
    internets = binary_yes_no(0.83)
    romantics = binary_yes_no(0.33)
    
    famrels = np.random.randint(1, 6, size=n_samples)
    freetimes = np.random.randint(1, 6, size=n_samples)
    goouts = np.random.randint(1, 6, size=n_samples)
    dalcs = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.70, 0.19, 0.07, 0.02, 0.02])
    walcs = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.38, 0.29, 0.20, 0.08, 0.05])
    healths = np.random.randint(1, 6, size=n_samples)
    
    absences = np.random.negative_binomial(1.5, 0.2, size=n_samples)
    absences = np.clip(absences, 0, 75)
    
    # Base performance logic tied to studytime, failures, absences, and parental education
    base_score = 10 + (studytimes * 1.5) - (failures * 2.5) - (absences * 0.25) + (medus * 0.5)
    
    g1 = np.clip(np.round(base_score + np.random.normal(0, 2, n_samples)), 0, 20).astype(int)
    g2 = np.clip(np.round(0.8 * g1 + 0.2 * base_score + np.random.normal(0, 1.5, n_samples)), 0, 20).astype(int)
    g3 = np.clip(np.round(0.85 * g2 + 0.15 * g1 + np.random.normal(0, 1.2, n_samples)), 0, 20).astype(int)
    
    df = pd.DataFrame({
        'school': schools, 'sex': sexes, 'age': ages, 'address': addresses,
        'famsize': famsizes, 'Pstatus': pstatuses, 'Medu': medus, 'Fedu': fedus,
        'Mjob': mjobs, 'Fjob': fjobs, 'reason': reasons, 'guardian': guardians,
        'traveltime': traveltimes, 'studytime': studytimes, 'failures': failures,
        'schoolsup': schoolsups, 'famsup': famsups, 'paid': paids, 'activities': activities,
        'nursery': nurseries, 'higher': highers, 'internet': internets, 'romantic': romantics,
        'famrel': famrels, 'freetime': freetimes, 'goout': goouts, 'Dalc': dalcs,
        'Walc': walcs, 'health': healths, 'absences': absences,
        'G1': g1, 'G2': g2, 'G3': g3
    })
    
    out_dir = os.path.dirname(__file__)
    csv_path = os.path.join(out_dir, 'student-por.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated dataset with {len(df)} records at {csv_path}")

if __name__ == "__main__":
    generate_uci_student_dataset()
