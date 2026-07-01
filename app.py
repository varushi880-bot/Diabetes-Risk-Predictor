from reportlab.platypus import SimpleDocTemplate,Paragraph,Image,Spacer,HRFlowable
from flask import session
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet 
from flask import send_file 
import os
import joblib
import numpy as np

from flask import Flask,render_template,request
app= Flask(__name__)
app.secret_key="arushi123"
model=joblib.load("model/diabetes_model.pkl")
scaler=joblib.load("model/scaler.pkl")
imputer=joblib.load("model/imputer.pkl")
print(model)
@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        print("POST REQUEST RECEIVED")
        Pregnancies=request.form["pregnancies"]
        Glucose=request.form["glucose"]
        BloodPressure=request.form["bloodpressure"]
        SkinThickness=request.form["skinthickness"]
        Insulin=request.form["insulin"]
        BMI=request.form["bmi"]
        dpf=request.form["dpf"]
        Age=request.form["age"]
        
        print("Pregnancies",Pregnancies)
        print("Glucose",Glucose)
        print("Blood Pressure",BloodPressure)
        print("SkinThickness",SkinThickness)
        print("Insulin",Insulin)
        print("BMI",BMI)
        print("dpf",dpf)
        print("Age",Age)
        
        data=np.array([[float(Pregnancies),float(Glucose),float(BloodPressure),float(SkinThickness),float(Insulin),float(BMI),float(dpf),float(Age)]])
        data=imputer.transform(data)
        data=scaler.transform(data)
        prediction=model.predict(data)
        probability=model.predict_proba(data)
                                        
        print("Prediction=",prediction)
        print("Data=",data)
        print("Probability=",probability)
        if prediction[0]==1:
           result="Diabetic"
           confidence=round(probability[0][1]*100,2)
           advice="Please consult a doctor and monitor your bload sugar regularly."
        else:
           result="Not Diabetic"
           confidence=round(probability[0][0]*100,2)
           advice="Keep maintaining a healthy lifestyle."
        if confidence >= 80:
               risk="High🔴"
        elif confidence>=50:
               risk="Medium🟡"
        else:
               risk="Low🟢"    
        session["result"]=result
        session["confidence"]=confidence
        session["risk"]=risk
        session["advice"]=advice      
        return render_template("index.html",result=result,confidence=confidence,advice=advice,risk=risk)
    return render_template("index.html")
@app.route("/download")
def download():
    current_time=datetime.now()
    formatted_time=current_time.strftime("%d-%m-%Y %I:%M %p")
    report_id=current_time.strftime("DIA-%Y%m%d-%H%M%S")
    result=session.get("result")
    confidence=session.get("confidence")
    risk=session.get("risk")
    advice=session.get("advice")
    doc=SimpleDocTemplate("report.pdf")
    styles=getSampleStyleSheet()
    story=[]
    logo=Image("static/logo.avif")
    logo.drawWidth=80
    logo.drawHeight=80
    story.append(logo)
    story.append(Spacer(1,10))
    story.append(Paragraph(f"<b>Report Generated:</b>{formatted_time}",styles["Normal"]))
    story.append(Paragraph(f"<b>Report ID:</b>{report_id}",styles["Normal"]))
    story.append(Spacer(1,10))
    story.append(Paragraph(f"<b>AI Diabetes Prediction Report</b>",styles["Title"]))
    story.append(HRFlowable())
    story.append(Spacer(1,10))
    story.append(Paragraph(f"<b>Prediction:</b>{result}",styles["Normal"]))
    story.append(Paragraph(f"<b>Confidence:</b>{confidence}%",styles["Normal"]))
    story.append(Paragraph(f"<b>Risk Level:</b>{risk}",styles["Normal"]))
    story.append(Paragraph(f"<b>Adice:</b>{advice}",styles["Normal"]))
    story.append(Paragraph(f"<br/> <br/>", styles["Normal"]))
    story.append(Paragraph("<b>Doctor Signature</b>",styles["Heading2"]))
    story.append(Paragraph("Dr. Rajesh Sharma ",styles["Normal"]))
    story.append(Paragraph("MD(General Physician)",styles["Normal"]))
    story.append(Paragraph("ABC Hospital",styles["Normal"]))
                 
    doc.build(story)
    return send_file("report.pdf",as_attachment=True)
if __name__=="__main__":
    app.run(debug=True)
    
    
    