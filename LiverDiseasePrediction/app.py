from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3
from datetime import datetime

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# optional translator
try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False

app = Flask(__name__)

# -------------------------
# Configuration / Model / DB
# -------------------------
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found in project folder. Train model and save as model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

DB_PATH = "records.db"
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)


# -------------------------
# DB Setup
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age REAL,
        gender INTEGER,
        total_bilirubin REAL,
        direct_bilirubin REAL,
        alk_phos REAL,
        alt REAL,
        ast REAL,
        total_proteins REAL,
        albumin REAL,
        ag_ratio REAL,
        prediction INTEGER,
        probability REAL,
        risk_score REAL,
        risk_level TEXT,
        stage TEXT,
        insights TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()


# -------------------------
# HOME ROUTE
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -------------------------
# DISEASE–STAGE + INSIGHTS LOGIC
# returns: disease, stage, tips, insights
# -------------------------
def get_disease_and_stage(values, prediction):
    tb = values[2]
    db = values[3]
    alp = values[4]
    alt = values[5]
    ast = values[6]
    tp = values[7]
    alb = values[8]
    agr = values[9]

    # If model or override says healthy
    if prediction == 0:
        tips = [
            "Maintain a balanced, healthy diet.",
            "Avoid excessive alcohol.",
            "Exercise regularly and stay hydrated."
        ]
        insights = "Liver parameters are within normal limits. Continue healthy lifestyle."
        return "No Liver Disease", "Normal", tips, insights

    # Fatty liver / mild inflammation
    if tb < 3 and (alt > 40 or ast > 40):
        tips = [
            "Reduce fried and sugary foods.",
            "Lose 5–7% body weight.",
            "Exercise at least 30 minutes daily.",
            "Avoid alcohol completely."
        ]
        insights = "Pattern suggests fatty liver or mild hepatocellular inflammation — lifestyle modification recommended."
        return "Fatty Liver (NAFLD)", "Mild", tips, insights

    # Alcoholic pattern (AST/ALT ratio > 2)
    if ast > alt and (ast / alt) >= 2:
        tips = [
            "STOP alcohol immediately.",
            "Take vitamin B complex only if prescribed by physician.",
            "High protein diet recommended.",
            "Regular LFT monitoring."
        ]
        insights = "AST > ALT with ratio >= 2 — pattern suggests alcoholic liver injury; immediate alcohol cessation advised."
        return "Alcoholic Liver Disease", "Moderate", tips, insights

    # Obstructive pattern
    if alp > 300 and tb > 2:
        tips = [
            "Possible bile duct obstruction — ultrasound recommended.",
            "Avoid fatty foods.",
            "Consult a gastroenterologist immediately."
        ]
        insights = "High ALP with elevated bilirubin suggests cholestasis or obstructive pathology; imaging recommended."
        return "Obstructive Liver Disease (Cholestasis)", "Moderate-Severe", tips, insights

    # Severe condition (possible cirrhosis)
    if alb < 3.5 and tb > 2.5 and alp > 250:
        tips = [
            "Consult a hepatologist urgently.",
            "Avoid salt and alcohol completely.",
            "High-protein diet needed; monitor for ascites."
        ]
        insights = "Low albumin with high bilirubin and ALP is concerning for cirrhosis or advanced liver disease."
        return "Cirrhosis (Liver Damage)", "Severe", tips, insights

    # Hepatitis pattern (very high transaminases)
    if (alt >= 100 or ast >= 100) and tb > 1.2:
        tips = [
            "Seek medical consultation.",
            "Avoid alcohol and painkillers.",
            "Follow antiviral treatment if required.",
            "Repeat LFT every month."
        ]
        insights = "Marked transaminase elevation with raised bilirubin — suggestive of acute or chronic hepatitis; further tests recommended."
        return "Hepatitis (A/B/C possible)", "Moderate-Severe", tips, insights

    # Default moderate inflammation
    tips = [
        "Avoid junk food and alcohol.",
        "Increase antioxidant-rich foods.",
        "Consult a doctor for enzyme analysis."
    ]
    insights = "Mild to moderate inflammation detected; monitor and re-evaluate after lifestyle changes or follow-up tests."
    return "Early Liver Inflammation", "Moderate", tips, insights


# -------------------------
# SAVE TO DATABASE
# -------------------------
def save_record(features, prediction, probability, risk_score, risk_level, stage, insights):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO records
        (age, gender, total_bilirubin, direct_bilirubin, alk_phos, alt, ast,
         total_proteins, albumin, ag_ratio, prediction, probability, risk_score, risk_level, stage, insights, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        features[0], int(features[1]), features[2], features[3], features[4],
        features[5], features[6], features[7], features[8], features[9],
        int(prediction), float(probability), float(risk_score), risk_level, stage, insights, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


# -------------------------
# PDF GENERATION (includes risk + insights)
# -------------------------
def generate_pdf(result_text, stage, tips, summary, features, risk_score, risk_level, insights, out_path="static/report.pdf"):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Liver Disease Prediction Report")

    # Meta
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 90, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Result and stage
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, height - 120, f"Result: {result_text}")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 140, f"Stage: {stage}")
    c.drawString(50, height - 160, f"Risk Score: {risk_score:.1f} ({risk_level})")

    # Preventive measures
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 190, "Preventive Measures:")
    c.setFont("Helvetica", 11)
    y = height - 210
    for tip in tips:
        c.drawString(70, y, f"- {tip}")
        y -= 16
        if y < 100:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)

    # Insights
    if y < 130:
        c.showPage()
        y = height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Doctor Insights:")
    c.setFont("Helvetica", 11)
    c.drawString(50, y - 30, insights)
    y -= 50

    # Summary
    if y < 140:
        c.showPage()
        y = height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Summary:")
    c.setFont("Helvetica", 11)
    c.drawString(50, y - 30, summary)
    y -= 60

    # Input values
    if y < 140:
        c.showPage()
        y = height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Input Values:")
    c.setFont("Helvetica", 11)
    labels = ["Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "ALP", "ALT", "AST", "Total Proteins", "Albumin", "A/G Ratio"]
    y -= 30
    for name, value in zip(labels, features):
        c.drawString(70, y, f"{name}: {value}")
        y -= 16
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)

    c.save()
    return out_path


# -------------------------
# PREDICTION ROUTE
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs (ensure HTML form fields are in the same order)
        features = [float(x) for x in request.form.values()]
        model_input = np.array(features).reshape(1, -1)

        # Raw model prediction
        prediction = int(model.predict(model_input)[0])

        # Probability if supported
        try:
            probability = float(model.predict_proba(model_input)[0][1])
        except Exception:
            probability = 1.0 if prediction == 1 else 0.0

        # Language selection (optional)
        lang = request.form.get("language", "en")

        # -------------------------
        # OVERRIDE: Force NORMAL if all clinical values are in normal ranges
        # -------------------------
        tb, db, alp, alt, ast, tp, alb, agr = features[2:10]
        if (
            tb <= 1.2 and db <= 0.3 and alp <= 150 and alt <= 40 and ast <= 40
            and 6 <= tp <= 8.3 and 3.5 <= alb <= 5.5 and 1 <= agr <= 2.5
        ):
            prediction = 0

        # -------------------------
        # Risk Score calculation
        # simple interpretable score — you can tweak weights
        # -------------------------
        # Normalize by common upper-limits to get comparable scale
        s_tb = (tb / 1.2) * 10
        s_alt = (alt / 40) * 10
        s_ast = (ast / 40) * 10
        risk_score = s_tb + s_alt + s_ast  # range roughly 0-60 in typical cases

        # Determine Risk Level properly
        if risk_score <= 20:
            risk_level = "Low Risk"
        elif risk_score <= 50:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"


        # Stage + disease logic + insights
        disease, stage, tips, insights = get_disease_and_stage(features, prediction)

        # Save record into DB
        save_record(features, prediction, probability, risk_score, risk_level, stage, insights)

        # Chart generation
        feature_names = [
            'Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphatase',
            'ALT', 'AST', 'Total Proteins', 'Albumin', 'A/G Ratio'
        ]
        normal_ranges = [1.2, 0.3, 150, 40, 40, 8.3, 5.5, 2.5]
        user_values = features[2:10]
        status = [1 if u > n else 0 for u, n in zip(user_values, normal_ranges)]

        plt.figure(figsize=(9, 5))
        colors = ['red' if s else 'green' for s in status]
        plt.bar(feature_names, user_values, color=colors)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Measured Values")
        plt.title("Liver Parameters vs Normal Ranges")
        for idx, thr in enumerate(normal_ranges):
            plt.plot([idx - 0.45, idx + 0.45], [thr, thr], 'k--', lw=0.7)
        plt.tight_layout()
        chart_path = os.path.join("static", "chart.png")
        plt.savefig(chart_path)
        plt.close()

        # Summary text
        abnormal = [n for n, s in zip(feature_names, status) if s == 1]
        summary = "All enzyme levels are within normal range." if not abnormal else "Abnormal levels detected in: " + ", ".join(abnormal)

        # Prepare display text
        result_text = (f"⚠️ Likely Liver Disease Detected: {disease}" if prediction == 1 else "✅ No Liver Disease Detected. Keep up your healthy habits!")
        prob_percent = f"{probability * 100:.1f}%"

        # Translation (if requested)
        translated_result = result_text
        translated_stage = stage
        translated_tips = tips
        translated_summary = summary
        translated_insights = insights

        if lang and lang != "en" and TRANSLATOR_AVAILABLE:
            try:
                translated_result = translator.translate(result_text, dest=lang).text
                translated_stage = translator.translate(stage, dest=lang).text
                translated_summary = translator.translate(summary, dest=lang).text
                translated_tips = [translator.translate(t, dest=lang).text for t in tips]
                translated_insights = translator.translate(insights, dest=lang).text
            except Exception:
                translated_result = result_text
                translated_stage = stage
                translated_tips = tips
                translated_summary = summary
                translated_insights = insights

        # Generate PDF using translated texts (so user gets report in selected language)
        pdf_path = generate_pdf(translated_result, translated_stage, translated_tips, translated_summary, features, risk_score, risk_level, translated_insights)

        # Render result page (we added risk_level & insights to template context)
        return render_template(
            "result.html",
            result=translated_result,
            stage=translated_stage,
            tips=translated_tips,
            chart_image=os.path.basename(chart_path),
            summary=translated_summary,
            probability=prob_percent,
            pdf_file=os.path.basename(pdf_path),
            risk_score=f"{risk_score:.1f}",
            risk_level=risk_level,
            insights=translated_insights
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------
# HISTORY ROUTE (all records)
# -------------------------
@app.route('/history')
def history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, created_at, age, gender, total_bilirubin, alt, ast, albumin, risk_score, risk_level, stage, prediction
        FROM records
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


# -------------------------
# DASHBOARD ROUTE
# -------------------------
@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM records")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM records WHERE prediction=1")
    diseased = cur.fetchone()[0]
    healthy = total - diseased

    cur.execute("""
        SELECT DATE(created_at), AVG(alt), AVG(ast)
        FROM records
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at)
    """)
    rows = cur.fetchall()

    dates = [row[0] for row in rows]
    avg_alts = [round(row[1], 2) if row[1] is not None else None for row in rows]
    avg_asts = [round(row[2], 2) if row[2] is not None else None for row in rows]

    cur.execute("""
        SELECT SUM(CASE WHEN total_bilirubin > 1.2 THEN 1 ELSE 0 END),
               SUM(CASE WHEN alt > 40 THEN 1 ELSE 0 END),
               SUM(CASE WHEN ast > 40 THEN 1 ELSE 0 END),
               SUM(CASE WHEN alk_phos > 150 THEN 1 ELSE 0 END)
        FROM records
    """)
    param_counts = cur.fetchone()

    conn.close()

    return render_template(
        "dashboard.html",
        total=total,
        diseased=diseased,
        healthy=healthy,
        dates=dates,
        avg_alts=avg_alts,
        avg_asts=avg_asts,
        param_counts=param_counts
    )


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
