import pandas as pd
import matplotlib.pyplot as plt	
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import fastf1

class F1Predictor:
    def get_last_valid_races(driver_code,year,num_races=3):
        schedule = fastf1.get_event_schedule(year,include_testing=False)
        recent_valid_races = []
        number=5
        
        for num in range(1,len(schedule)):
            event = schedule.loc[schedule['RoundNumber']==len(schedule)-num]
            eventName = event['EventName'].values[0]
            eventDate = event['Session1Date']
            event_date_value = pd.to_datetime(event['EventDate'].values[0]).tz_localize(None)
            if event_date_value>datetime.now():
                print("Didn't Happen did it")
                continue
            print("EVENT NAME ISSSS", eventName)
            try:
                race = fastf1.get_session(year,eventName,'R')
                race.load(telemetry = False, weather= False, laps=False,messages=False)
                results = race.results.query(f"Abbreviation == '{driver_code.upper()}' ")
                if not results.empty:
                    status = results.iloc[0]['Status']
                    if 'Finished' in status or 'Lap' in status:
                        gridpos=results.iloc[0]['GridPosition']
                        pos=results.iloc[0]['Position']
                        recent_valid_races.append(eventName)
                        recent_valid_races.append(gridpos)
                        recent_valid_races.append(pos)
                        
                        
                        if len(recent_valid_races)>=9:
                            number = num
                            break

                else:
                    print("RESULT EMPTYYYYYY")
                    continue
            except:
                continue
        try:
            event= schedule.loc[schedule['RoundNumber']==len(schedule)-number+3]
            eventName = event['EventName'].values[0]
            recent_valid_races.append(eventName)
        except IndexError:
            track4="Unknown"
            recent_valid_races.append(track4)
        columns = ['Track1','Grid1','Finish1','Track2','Grid2','Finish2','Track3','Grid3','Finish3','Track4']
        return pd.DataFrame([recent_valid_races],columns=columns)
    
      
    def generate_training_data(driver_code,year,num_races=3):
        training_data=[]
        schedule = fastf1.get_event_schedule(year,include_testing=False)

        for num in range(3,len(schedule)):
            features = []
            valid=True
            event = schedule.iloc[num]
            eventName = event['EventName']
            event_date_value = pd.to_datetime(event['EventDate']).tz_localize(None)
            if event_date_value>datetime.now():
                print("Didn't Happen did it")
                continue
            for i in range(num-num_races,num):
                previous_event = schedule.iloc[i]
                try:
                    session = fastf1.get_session(year,previous_event['EventName'],'R')
                    session.load(telemetry=False,laps=False,weather=False,messages=False)
                    results = session.results.query(f"Abbreviation == '{driver_code.upper()}'")
                    if results.empty:
                        valid=False
                        break
                    status = results.iloc[0]['Status'] 
                    if 'Finished' not in status and 'Lap' not in status:
                        valid=False
                        break
                    features.append(previous_event['EventName'])
                    features.append(results.iloc[0]['GridPosition'])
                    features.append(results.iloc[0]['Position'])

                except:
                    valid=False
                    break
            if not valid:
                continue
                
            try:
                session = fastf1.get_session(year,eventName,'R')
                session.load(telemetry=False,laps=False,weather=False,messages=False)
                results = session.results.query(f"Abbreviation == '{driver_code.upper()}'")
                if results.empty:
                    valid=False
                    continue
                status = results.iloc[0]['Status']
                if 'Finished' not in status and 'Lap' not in status:
                    valid=False
                    continue
                features.append(eventName)
                features.append(results.iloc[0]['GridPosition'])
                features.append(results.iloc[0]['Position'])
                training_data.append(features)
                
            except:
                continue
        columns = ['Track1','Grid1','Finish1','Track2','Grid2','Finish2','Track3','Grid3','Finish3','Track4','Grid4','TargetFinish']
        
        return pd.DataFrame(training_data,columns=columns)
    
    def classify_position(pos):
        if(pos<=3):
            return "Podium"
        elif(pos<=10):
            return "Points"
        elif(pos<=20):
            return "No Points"
        else:
            return "Unkown"
        
training_data1 = F1Predictor.generate_training_data('Ham',2024)
training_data1=training_data1.dropna()
training_data2 = F1Predictor.generate_training_data('Ham',2023)
training_data2=training_data2.dropna()
training_data3 = F1Predictor.generate_training_data('Ham',2022)
training_data3=training_data3.dropna()
training_data4 = F1Predictor.generate_training_data('Ham',2021)
training_data4=training_data4.dropna()
training_data5 = F1Predictor.generate_training_data('Ham',2020)
training_data5=training_data5.dropna()
training_data6 = F1Predictor.generate_training_data('Ham',2019)
training_data6=training_data6.dropna()
training_data7 = F1Predictor.generate_training_data('Ham',2018)
training_data7=training_data7.dropna()



training_data = pd.concat([training_data1,training_data2,training_data3,training_data4,training_data5,training_data6,training_data7],ignore_index=True)
training_data['TargetFinish']=training_data['TargetFinish'].apply(F1Predictor.classify_position)
print("TRAINING SHAPE: ",training_data)
for col in ['Finish1', 'Finish2', 'Finish3']:
    training_data[col] = training_data[col].apply(F1Predictor.classify_position)

Xtrain= training_data.drop(columns=['TargetFinish'])
Ytrain= training_data['TargetFinish']
#XTest = F1Predictor.get_last_valid_races('Ha',2025)
tracks = ['Track1','Track2','Track3','Track4','Finish1','Finish2','Finish3']
Xtrain[tracks] = Xtrain[tracks].fillna('Unknown')
#XTest[tracks] = XTest[tracks].fillna('Unknown')
Xtrain = pd.get_dummies(Xtrain, columns=tracks)

print("YOOOOO LOOK AT ME")
#print(XTest)
#XTest = XTest.dropna()
model = RandomForestClassifier()
#if XTest.isnull().values.any():
    #print("NULL VALUES")
if Xtrain.isnull().values.any():
    print("NULL VALUESSSSS")
print(f"Xtrain shape: {Xtrain.shape}")
print(f"Ytrain shape: {Ytrain.shape}")

Xtrains, X_val, Ytrains, y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=42)
model.fit(Xtrains,Ytrains)
Ypred = model.predict(X_val)
accuracy = accuracy_score(Ypred,y_val)
print("Training Accuracy:",accuracy)
#YTest= model.predict(XTest)
#print("Predicted Position:", YTest) 
            
        





                



    


