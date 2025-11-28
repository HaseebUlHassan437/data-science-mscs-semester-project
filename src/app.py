# Simple command-line interface for heart disease prediction
import sys
sys.path.append('..')
from predict import load_model, predict_heart_disease

def get_patient_input():
    """Get patient data from user"""
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTION SYSTEM")
    print("="*60)
    print("\nEnter patient information:")
    
    patient = {}
    
    # Get inputs
    patient['age'] = float(input("Age (years): "))
    patient['sex'] = float(input("Sex (1=Male, 0=Female): "))
    patient['cp'] = float(input("Chest Pain Type (1-4): "))
    patient['trestbps'] = float(input("Resting Blood Pressure (mm Hg): "))
    patient['chol'] = float(input("Cholesterol (mg/dl): "))
    patient['fbs'] = float(input("Fasting Blood Sugar >120 (1=Yes, 0=No): "))
    patient['restecg'] = float(input("Resting ECG (0-2): "))
    patient['thalach'] = float(input("Max Heart Rate: "))
    patient['exang'] = float(input("Exercise Induced Angina (1=Yes, 0=No): "))
    patient['oldpeak'] = float(input("ST Depression: "))
    patient['slope'] = float(input("Slope (1-3): "))
    patient['ca'] = float(input("Number of Major Vessels (0-3): "))
    patient['thal'] = float(input("Thalassemia (3/6/7): "))
    
    # Engineered features
    patient['age_above_50'] = 1 if patient['age'] > 50 else 0
    patient['high_chol'] = 1 if patient['chol'] > 200 else 0
    patient['low_heart_rate'] = 1 if patient['thalach'] < 140 else 0
    
    return patient

def main():
    """Main application"""
    try:
        # Load model
        print("\nLoading model...")
        model = load_model()
        print("Model loaded successfully!")
        
        while True:
            # Get patient data
            patient_data = get_patient_input()
            
            # Make prediction
            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            
            prediction, probability = predict_heart_disease(model, patient_data)
            
            if prediction == 1:
                print(f"\nResult: HEART DISEASE DETECTED")
                print(f"Confidence: {probability[1]*100:.2f}%")
                print("\nRecommendation: Consult a cardiologist immediately")
            else:
                print(f"\nResult: NO HEART DISEASE")
                print(f"Confidence: {probability[0]*100:.2f}%")
                print("\nRecommendation: Maintain healthy lifestyle")
            
            print("="*60)
            
            # Ask for another prediction
            again = input("\nPredict for another patient? (y/n): ")
            if again.lower() != 'y':
                print("\nThank you for using the system!")
                break
                
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()